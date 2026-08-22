"""CogriaAgent orchestration service — build_app() wires the kernel to the seams.

Runs as a private service (e.g. 127.0.0.1:8001), called only by the BFF. The BFF
puts a JWT (minted by the business backend's /agent-auth/exchange) in the
Authorization header; we verify it locally (shared HS256 secret). No tenant
dimension — single-tenant by design.

Nothing here is domain-specific: catalog, persistence and execution all sit
behind the injected seams in protocols.py, and app name / issuer / required
claims / model / prompt come from AgentConfig.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from . import summarizer
from .artifacts import artifact_tool_names, build_artifact_tools
from .attachments import (
    ATTACHMENT_GUIDANCE,
    AttachmentRejected,
    Budget,
    build_turn_content,
    hydrate_history,
)
from .config import AgentConfig
from .estimate import estimate_messages, warm_encoder
from .graph import build_graph, history_to_messages
from .llm import ConfigSystemPromptProvider, DefaultLLMFactory
from .protocols import (
    TITLE_MAX_CHARS,
    ActionExecutor,
    AttachmentRepository,
    AttachmentStore,
    CatalogProvider,
    ConversationBackend,
    DocumentExtractor,
    SystemPromptProvider,
)
from .prune import prune_result
from .recall import RECALL_GUIDANCE, build_recall_tool
from .sse import extract_proposal, sse
from .tools import build_tools_from_catalog

logger = logging.getLogger("cogria.agentserv")

_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _subscribe(run: dict[str, Any]) -> tuple[asyncio.Queue | None, list[bytes]]:
    """Attach to a run and snapshot what it has already emitted.

    Both halves happen here, with no `await` between them, and that is
    load-bearing: `emit()` appends to `events` and pushes to every queue
    synchronously, so a suspension point between subscribing and snapshotting
    would let a frame land in both — the client would see it twice. Returning
    them together makes it impossible to interleave anything at the call site.

    A finished run yields `(None, backlog)`: there is nothing left to follow, so
    the caller just replays and stops.
    """
    if run["done"]:
        return None, list(run["events"])
    q: asyncio.Queue = asyncio.Queue()
    run["subs"].add(q)
    return q, list(run["events"])


def build_app(
    config: AgentConfig,
    *,
    conversation_backend: ConversationBackend,
    action_executor: ActionExecutor,
    catalog_provider: CatalogProvider,
    llm_factory: Any | None = None,
    prompt_provider: SystemPromptProvider | None = None,
    attachment_store: AttachmentStore | None = None,
    attachment_repo: AttachmentRepository | None = None,
    document_extractor: DocumentExtractor | None = None,
) -> FastAPI:
    """Construct the FastAPI app, binding the four required seams (+ optional
    LLM/prompt providers, which default to config-driven implementations).

    Passing `attachment_store` + `attachment_repo` switches on file uploads: the
    /attachments routes get mounted and /chat starts accepting `attachment_ids`.
    Omit them and the service behaves exactly as it did before attachments
    existed. `document_extractor` defaults to DefaultDocumentExtractor.
    """

    llm_factory = llm_factory or DefaultLLMFactory(config)
    prompt_provider = prompt_provider or ConfigSystemPromptProvider(config)

    attachment_service = None
    if attachment_store is not None and attachment_repo is not None:
        from .attachments import AttachmentService, DefaultDocumentExtractor

        attachment_service = AttachmentService(
            store=attachment_store,
            repo=attachment_repo,
            extractor=document_extractor
            or DefaultDocumentExtractor(
                timeout_seconds=config.attachments.extract_timeout_seconds,
                max_uncompressed_bytes=config.attachments.max_uncompressed_bytes,
                max_zip_entries=config.attachments.max_zip_entries,
            ),
            config=config.attachments,
        )
    elif attachment_store is not None or attachment_repo is not None:
        raise ValueError("attachments need BOTH attachment_store and attachment_repo")

    # Validate here rather than on the first oversized result: a budget that
    # cannot shrink anything is a deployment mistake, and mid-turn is the worst
    # possible moment to discover it.
    config.prune.validated()
    # Resolve the token estimator now, off the request path.
    warm_encoder()

    # Attachment content is re-attached at request time, not stored on the
    # message, so pricing a replay set needs the injection budget. `None` when
    # uploads are off, because then nothing is ever hydrated.
    attachment_pricing = config.attachments if attachment_service is not None else None

    if not config.auth.jwt_secret:
        logger.warning("JWT_SECRET is empty; JWT verification will reject all requests.")

    app = FastAPI(
        title=config.app_name,
        version="0.0.0",
        docs_url="/docs" if config.env == "dev" else None,
        redoc_url=None,
    )

    def verify_jwt(authorization: str = Header(default="")) -> dict:
        if not config.auth.jwt_secret:
            raise HTTPException(500, "JWT secret not configured")
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        token = authorization[7:]
        try:
            return jwt.decode(
                token,
                config.auth.jwt_secret,
                algorithms=["HS256"],
                issuer=config.auth.jwt_issuer,
                options={"require": config.auth.required_claims},
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(401, "token expired")
        except jwt.InvalidIssuerError:
            raise HTTPException(401, "invalid issuer")
        except jwt.InvalidTokenError as e:
            raise HTTPException(401, f"invalid token: {e}")

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "service": config.app_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attachments": attachment_service is not None,
        }

    if attachment_service is not None:
        from .attachments import build_attachment_router

        app.include_router(build_attachment_router(attachment_service, verify_jwt))

    @app.get("/debug/whoami")
    def whoami(claims: dict = Depends(verify_jwt)) -> dict:
        return {"sub": claims.get("sub"), "role": claims.get("role"), "locale": claims.get("locale")}

    # Runs in flight, keyed by conversation id. Lets a client that navigated
    # away re-attach (GET /conversations/{id}/stream) and replay what it missed.
    # In-process state: it assumes a single worker. Behind several workers the
    # POST and the resume GET can land on different processes, and the resume
    # degrades to "no active run" — correct, just not live.
    active_runs: dict[str, dict[str, Any]] = {}

    async def owned_meta(conversation_id: str, claims: dict) -> dict:
        """The one ownership gate every conversation route runs. Someone else's
        conversation, a nonexistent one and a soft-deleted one all return the
        same 404 — an id must not be probeable for existence."""
        meta = await conversation_backend.fetch_meta(conversation_id)
        if not meta or meta.get("user_id") != claims.get("sub") or meta.get("deleted_at"):
            raise HTTPException(404, "conversation not found")
        return meta

    @app.get("/conversations")
    async def list_conversations(
        claims: dict = Depends(verify_jwt), limit: int = 50, offset: int = 0
    ) -> dict:
        """The caller's own history. Scoped by the `sub` claim and never by a
        caller-supplied user id — the token is the only identity."""
        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))
        rows = await conversation_backend.list_conversations(
            user_id=claims.get("sub"), limit=limit, offset=offset
        )
        return {"conversations": rows}

    @app.get("/conversations/{conversation_id}")
    async def get_conversation(
        conversation_id: str, claims: dict = Depends(verify_jwt)
    ) -> dict:
        """Full message history for replay, plus whether a reply is being
        generated right now — the flag the client uses to decide to re-attach."""
        meta = await owned_meta(conversation_id, claims)
        messages = await conversation_backend.fetch_messages_full(conversation_id)
        # What the NEXT request would cost, against the window it has to fit in.
        # The stored `total_*_tokens` are lifetime billing figures and grow with
        # every turn, so they answer a different question entirely.
        replay = await conversation_backend.fetch_history(conversation_id, for_llm=True)
        window = config.summarizer.context_window
        pressure = estimate_messages(replay, attachments=attachment_pricing)
        return {
            "conversation": {
                **meta,
                "estimated_context_tokens": pressure,
                "context_window": window,
                "context_usage_ratio": round(pressure / window, 4) if window else None,
            },
            "messages": messages,
            "active": conversation_id in active_runs,
        }

    @app.patch("/conversations/{conversation_id}")
    async def rename_conversation(
        conversation_id: str, request: Request, claims: dict = Depends(verify_jwt)
    ) -> dict:
        """Rename a conversation. An empty/blank title clears it, which puts the
        row back on the title derived from its opening message."""
        await owned_meta(conversation_id, claims)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "invalid json") from None
        if not isinstance(body, dict):
            raise HTTPException(422, "body must be a JSON object")
        raw = body.get("title")
        if raw is not None and not isinstance(raw, str):
            raise HTTPException(422, "title must be a string")
        title = " ".join((raw or "").split())[:TITLE_MAX_CHARS] or None
        await conversation_backend.rename_conversation(conversation_id, title=title)
        return {"ok": True, "title": title}

    @app.delete("/conversations/{conversation_id}")
    async def delete_conversation(
        conversation_id: str, claims: dict = Depends(verify_jwt)
    ) -> dict:
        """Soft delete: it leaves the owner's history but the row and its
        messages survive for support and audit."""
        await owned_meta(conversation_id, claims)
        await conversation_backend.soft_delete_conversation(conversation_id)
        return {"ok": True}

    @app.get("/conversations/{conversation_id}/stream")
    async def resume_conversation_stream(
        conversation_id: str, claims: dict = Depends(verify_jwt)
    ) -> StreamingResponse:
        """Re-attach to a run in progress: replay every frame it has emitted so
        far, then follow it live until done. 404 when nothing is running — the
        turn finished while the client was away, so it refetches history."""
        await owned_meta(conversation_id, claims)
        run = active_runs.get(conversation_id)
        if run is None:
            raise HTTPException(404, "no active run")
        q, backlog = _subscribe(run)

        async def event_stream():
            try:
                # No `ready` here: the client already has its tool list, and this
                # is a continuation of a turn rather than the start of one.
                yield sse("conversation", {"conversation_id": conversation_id, "new": False})
                for frame in backlog:
                    yield frame
                while q is not None:
                    item = await q.get()
                    if item is None:
                        break
                    yield item
            finally:
                if q is not None:
                    run["subs"].discard(q)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/chat")
    async def chat(request: Request, claims: dict = Depends(verify_jwt)) -> StreamingResponse:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "invalid json")

        user_message = (body.get("message") or "").strip()
        if not user_message:
            raise HTTPException(422, "message is required")

        conversation_id = body.get("conversation_id")
        if conversation_id is not None:
            # Public ids are opaque strings; an unknown or malformed one simply
            # fails the ownership lookup below as a 404.
            conversation_id = str(conversation_id).strip() or None

        # Propose/confirm follow-up (MVP-A): the [CONFIRMED] turn carries the
        # proposal_token; feed the LLM an augmented message so it re-supplies the
        # token, while the persisted user row keeps the clean keyword.
        proposal_token = (body.get("proposal_token") or "").strip()
        llm_message = user_message
        if proposal_token and user_message == "[CONFIRMED]":
            llm_message = f"[CONFIRMED] proposal_token={proposal_token}"
        elif user_message == "[CANCELLED]":
            llm_message = "[CANCELLED]"

        locale = claims.get("locale")
        owner_sub = str(claims.get("sub") or "")

        # Files the user attached to THIS message (uploaded beforehand via
        # /attachments). Resolving proves they exist and belong to the caller.
        attachment_ids = [str(i) for i in (body.get("attachment_ids") or []) if i]
        if attachment_ids and attachment_service is None:
            raise HTTPException(501, "attachments are not enabled on this server")
        records: list[dict[str, Any]] = []
        if attachment_ids:
            try:
                records = await attachment_service.resolve(attachment_ids, owner_sub=owner_sub)
            except AttachmentRejected as e:
                raise HTTPException(e.status_code, e.message) from e

        # Artifact tools come from the front-end registry via the BFF; advertised
        # to the LLM but dispatched as no-ops (the tool_call event IS the render).
        artifact_defs = body.get("artifact_tools") or []
        artifact_names = artifact_tool_names(artifact_defs)

        # Per-request identity for the executor (the LLM never sees this).
        bearer = request.headers.get("authorization", "")
        req_context = {"claims": claims, "bearer": bearer}

        # What actually gets persisted for the user's turn: the clean text plus
        # a lightweight reference to each file (the extracted text stays in the
        # attachment record — one copy, not one per message).
        user_content: dict[str, Any] = {"text": user_message}
        if records:
            user_content["attachments"] = [
                {
                    "id": r["id"],
                    "name": r["filename"],
                    "mime": r["mime"],
                    "size": r["size_bytes"],
                    "kind": r["kind"],
                }
                for r in records
            ]

        is_new = conversation_id is None
        if is_new:
            conversation_id = await conversation_backend.create_conversation(
                first_message=user_message,
                model=llm_factory.model_name(),
                user_id=owner_sub or None,
                first_content=user_content,
            )
            prior: list[dict[str, Any]] = []
        else:
            # Ownership gate on continuation, before any graph or LLM work.
            # Without it any authenticated caller could resume someone else's
            # conversation — feeding its history to the model (a read) and
            # appending to it (a write).
            await owned_meta(conversation_id, claims)
            prior = await conversation_backend.fetch_history(conversation_id, for_llm=True)
            # Persist the user's message NOW rather than at flush: if the client
            # navigates away mid-run, a history fetch must still show this turn.
            await conversation_backend.append_messages(
                conversation_id,
                messages=[{"role": "user", "content": user_content}],
                usage=None,
                model=None,
            )

        # One character budget for the whole request, spent on this turn first
        # so the newest document always gets its full allowance.
        turn_content: Any = llm_message
        if attachment_service is not None:
            budget = Budget(config.attachments.max_chars_total, config.attachments.max_chars_per_doc)
            vision_enabled = attachment_service.vision_enabled
            turn_content = await build_turn_content(
                text=llm_message,
                records=records,
                service=attachment_service,
                budget=budget,
                vision_enabled=vision_enabled,
            )
            if prior:
                prior = await hydrate_history(
                    prior,
                    service=attachment_service,
                    config=config.attachments,
                    budget=budget,
                    vision_enabled=vision_enabled,
                    owner_sub=owner_sub,
                )

        initial_messages = history_to_messages(prior)
        initial_messages.append(HumanMessage(content=turn_content))

        # Images (this turn or replayed) require the configured vision model;
        # everything else stays on the default one.
        model_override = (
            config.attachments.vision_model if _contains_image(initial_messages) else None
        )
        prompt_suffix = ATTACHMENT_GUIDANCE if (records or _has_attachments(prior)) else ""

        catalog = await catalog_provider.get_catalog()
        tools = build_tools_from_catalog(catalog, executor=action_executor, context=req_context)
        tools = tools + build_artifact_tools(artifact_defs)
        if config.recall.enabled:
            # Bound to THIS conversation, so the model cannot address another one.
            tools = tools + [
                build_recall_tool(
                    backend=conversation_backend,
                    conversation_id=conversation_id,
                    config=config.recall,
                )
            ]
            prompt_suffix = prompt_suffix + RECALL_GUIDANCE

        # The exact prompt the graph will send. Compaction replays it verbatim so
        # its request is a prefix of this one and the provider's cache is reused.
        system_prompt_text = prompt_provider.system_prompt(locale=locale) + prompt_suffix

        graph = build_graph(
            tools,
            llm_factory=llm_factory,
            prompt_provider=prompt_provider,
            locale=locale,
            max_turns=config.graph.max_turns,
            max_tool_calls=config.graph.max_tool_calls_per_turn,
            prompt_suffix=prompt_suffix,
            model_override=model_override,
        )

        logger.info(
            "chat user=%s conv=%s new=%s tools=%s prior=%d attachments=%d model=%s",
            claims.get("sub"), conversation_id, is_new, [t.name for t in tools], len(prior),
            len(records), model_override or llm_factory.model_name(),
        )

        acc: dict[str, Any] = {
            "assistant_text": "",
            "pending_calls": {},
            "tool_results": [],
            "resolved_call_ids": set(),
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_reason": None,
            "stream_error": None,
            "flushed": False,
        }

        async def flush_turn() -> None:
            """Persist the accumulated turn exactly once, decoupled from the
            client (a disconnect cancels the generator, but the reply must land)."""
            if acc["flushed"]:
                return
            acc["flushed"] = True
            try:
                await _persist_turn(
                    backend=conversation_backend,
                    conversation_id=conversation_id,
                    acc=acc,
                    artifact_names=artifact_names,
                    model=llm_factory.model_name(),
                    prune_config=config.prune,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("persist turn failed conv=%s: %s", conversation_id, e)

            if config.summarizer.enabled:
                try:
                    task = asyncio.ensure_future(
                        summarizer.maybe_summarize(
                            backend=conversation_backend,
                            llm_factory=llm_factory,
                            config=config.summarizer,
                            conversation_id=conversation_id,
                            # Handed the same prompt and tools the conversation
                            # itself uses, so the summarization call can be a
                            # genuine prefix of the request just sent and reuse
                            # the provider's warm cache.
                            system_prompt=system_prompt_text,
                            tools=tools,
                            attachments=attachment_pricing,
                        )
                    )
                    # Hold a reference: a bare create_task is only weakly held by
                    # the loop, so a compaction could be collected mid-flight.
                    _BACKGROUND_TASKS.add(task)
                    task.add_done_callback(_BACKGROUND_TASKS.discard)
                except Exception:  # noqa: BLE001
                    pass

        # The graph runs in its own task, decoupled from any one SSE response: a
        # client disconnect (navigating away) cancels only that subscriber,
        # while the turn runs to completion and persists via flush_turn. Every
        # frame is kept in the run's backlog so a client that comes back can
        # replay what it missed and follow the rest live.
        run: dict[str, Any] = {"events": [], "subs": set(), "done": False}
        active_runs[conversation_id] = run

        def emit(frame: bytes) -> None:
            run["events"].append(frame)
            # Snapshot the set: a subscriber whose generator is being torn down
            # can discard itself while we iterate.
            for sub in list(run["subs"]):
                sub.put_nowait(frame)

        async def rebuild_after_compaction() -> list[Any]:
            """Re-derive this turn's messages from the freshly compacted history.

            This turn's user message is already persisted — a continuation
            appends it before the graph starts, a new conversation stores it at
            creation — so the refetched history already contains it and must not
            have it appended a second time.
            """
            rows = await conversation_backend.fetch_history(conversation_id, for_llm=True)
            if attachment_service is not None:
                # A fresh budget: the original one was spent on the request that
                # just failed.
                rows = await hydrate_history(
                    rows,
                    service=attachment_service,
                    config=config.attachments,
                    budget=Budget(
                        config.attachments.max_chars_total,
                        config.attachments.max_chars_per_doc,
                    ),
                    vision_enabled=attachment_service.vision_enabled,
                    owner_sub=owner_sub,
                )
            return history_to_messages(rows)

        async def drive(messages_for_run: list[Any]) -> None:
            """Stream one model run into `acc`. Exceptions propagate: the caller
            decides whether a failure is recoverable."""
            async for event in graph.astream_events(
                {"messages": messages_for_run, "turns": 0}, version="v2"
            ):
                kind = event.get("event")
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        acc["assistant_text"] += chunk.content
                        emit(sse("text", {"delta": chunk.content}))
                elif kind == "on_chat_model_end":
                    out = event["data"].get("output")
                    if isinstance(out, (AIMessage, AIMessageChunk)):
                        usage = getattr(out, "usage_metadata", None) or {}
                        acc["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                        acc["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
                        # The provider's own verdict on how generation
                        # ended. Inferring it locally hid `length`
                        # (output truncated) behind a `tool_calls` that
                        # looked perfectly healthy. Last round wins.
                        finish = (getattr(out, "response_metadata", None) or {}).get(
                            "finish_reason"
                        )
                        acc["finish_reason"] = finish or None
                        if finish == "length":
                            # Truncated mid-array: the graph refuses to
                            # run these calls, so don't announce them
                            # either (the panel would show cards that
                            # never resolve). Tell the client instead —
                            # the streamed text has already gone out and
                            # must not be read as a complete answer.
                            emit(sse("truncated", {"finish_reason": "length"}))
                            continue
                        for call in getattr(out, "tool_calls", []) or []:
                            acc["pending_calls"][call["id"]] = {
                                "name": call["name"],
                                "args": call.get("args", {}),
                            }
                            emit(sse(
                                "tool_call",
                                {"id": call["id"], "name": call["name"], "args": call.get("args", {})},
                            ))
                elif kind == "on_tool_end":
                    output = event["data"].get("output")
                    name = event.get("name") or ""
                    content = output.content if isinstance(output, ToolMessage) else str(output)
                    # Tools are invoked with the whole tool_call, so the
                    # ToolMessage names its own id. Pair on that: the old
                    # by-name-in-arrival-order fallback silently mispairs
                    # as soon as any call is deduped or capped away, and
                    # survives only for tools returning a bare value.
                    tool_call_id = (
                        output.tool_call_id if isinstance(output, ToolMessage) else None
                    )
                    if tool_call_id:
                        acc["resolved_call_ids"].add(tool_call_id)
                    else:
                        for cid, c in acc["pending_calls"].items():
                            if c["name"] == name and cid not in acc["resolved_call_ids"]:
                                tool_call_id = cid
                                acc["resolved_call_ids"].add(cid)
                                break
                    acc["tool_results"].append(
                        {"tool_call_id": tool_call_id, "name": name, "content": content}
                    )
                    proposal = extract_proposal(content)
                    if proposal is not None:
                        emit(sse(
                            "confirm_required",
                            {
                                "proposal_token": proposal.get("proposal_token"),
                                "summary": proposal.get("summary", ""),
                                "action_name": name,
                            },
                        ))
                    elif name in artifact_names:
                        pass  # the tool_call event already drove the panel
                    else:
                        emit(sse("tool_result", {"name": name, "content": content}))
        async def run_graph():
            # A provider that rejects the request as too long is telling us
            # something compaction can act on. Retry only while nothing has been
            # emitted yet — once text or a tool call has gone out to the client,
            # re-running would duplicate it.
            attempt = 0
            messages_for_run = initial_messages
            try:
                while True:
                    try:
                        await drive(messages_for_run)
                        break
                    except Exception as e:  # noqa: BLE001
                        recoverable = (
                            attempt < config.graph.max_overflow_retries
                            and _is_context_overflow(e)
                            and not acc["assistant_text"]
                            and not acc["pending_calls"]
                        )
                        if recoverable:
                            attempt += 1
                            logger.warning(
                                "context overflow conv=%s; compacting and retrying (%d/%d)",
                                conversation_id, attempt, config.graph.max_overflow_retries,
                            )
                            emit(sse("compacting", {"reason": "context_overflow"}))
                            compacted = await summarizer.compact(
                                backend=conversation_backend,
                                llm_factory=llm_factory,
                                config=config.summarizer,
                                conversation_id=conversation_id,
                                system_prompt=system_prompt_text,
                                tools=tools,
                                attachments=attachment_pricing,
                                force=True,
                            )
                            if compacted:
                                messages_for_run = await rebuild_after_compaction()
                                continue
                            # Nothing safe left to fold: the original error is
                            # the honest thing to report.
                            logger.warning(
                                "context overflow conv=%s: no compactable history",
                                conversation_id,
                            )
                        logger.exception("chat stream failed: %s", e)
                        acc["stream_error"] = f"{type(e).__name__}: {e}"
                        emit(sse("error", {"message": acc["stream_error"]}))
                        break
            finally:
                # Persist BEFORE emitting done: when a client sees the turn
                # finish, a history fetch must already include it.
                await flush_turn()
                emit(sse("done", {"conversation_id": conversation_id}))
                run["done"] = True
                for sub in list(run["subs"]):
                    sub.put_nowait(None)  # sentinel: stop following
                # Only retract our own entry. A second turn on this conversation
                # may already have replaced it, and that one is still live.
                if active_runs.get(conversation_id) is run:
                    del active_runs[conversation_id]

        graph_task = asyncio.ensure_future(run_graph())
        _BACKGROUND_TASKS.add(graph_task)
        graph_task.add_done_callback(_BACKGROUND_TASKS.discard)

        async def event_stream():
            yield sse("ready", {"tools": [t.name for t in tools]})
            yield sse("conversation", {"conversation_id": conversation_id, "new": is_new})
            # Subscribing here rather than before the response means the graph
            # task may already have emitted; the backlog covers exactly that gap,
            # including the case where the whole turn finished first.
            q, backlog = _subscribe(run)
            try:
                for frame in backlog:
                    yield frame
                while q is not None:
                    item = await q.get()
                    if item is None:
                        break
                    yield item
            finally:
                if q is not None:
                    run["subs"].discard(q)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    return app


#: Substrings that identify a provider rejecting a request for being too long.
#: Every OpenAI-compatible gateway words this differently and none of them use a
#: shared status code, so matching text is the only portable signal. Kept
#: deliberately narrow: a false positive costs a pointless compaction plus a
#: retry, so nothing generic like "token" or "limit" belongs here.
_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "reduce the length of the messages",
    "prompt is too long",
    "input length and `max_tokens` exceed",
    "too many tokens",
)


def _is_context_overflow(error: BaseException) -> bool:
    """True when a request failed because the prompt did not fit.

    Walks the exception chain: the SDK error that names the reason is usually
    wrapped by the time LangChain re-raises it.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        haystack = f"{getattr(current, 'code', '') or ''} {current}".lower()
        if any(marker in haystack for marker in _OVERFLOW_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _contains_image(messages: list[Any]) -> bool:
    """True if any message carries a multimodal image block — the signal to
    route this request to the vision model."""
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "image_url" for b in content
        ):
            return True
    return False


def _has_attachments(rows: list[dict[str, Any]]) -> bool:
    return any((r.get("content") or {}).get("attachments") for r in rows)


async def _persist_turn(
    *,
    backend: ConversationBackend,
    conversation_id: Any,
    acc: dict[str, Any],
    artifact_names: set[str],
    model: str | None,
    prune_config: Any | None = None,
) -> None:
    """Assemble the turn's assistant/tool messages and append them. The user
    message is already persisted (create_conversation for a new conversation,
    an up-front append for a continuation)."""
    messages: list[dict[str, Any]] = []

    pending_calls = acc["pending_calls"]
    assistant_content: dict[str, Any] = {"text": acc["assistant_text"]}
    tool_calls_meta: list[dict[str, Any]] = []
    if pending_calls:
        assistant_content["tool_calls"] = [
            {"id": cid, "name": c["name"], "args": c["args"]} for cid, c in pending_calls.items()
        ]
        tool_calls_meta = [
            {
                "tool_call_id": cid,
                "tool_name": c["name"],
                "tool_type": "artifact" if c["name"] in artifact_names else "action",
                "args": c["args"],
                "status": "success",
            }
            for cid, c in pending_calls.items()
        ]

    stream_error = acc["stream_error"]
    # Prefer what the provider reported; infer only when it told us nothing.
    # The inferred value can express stop/tool_calls and nothing else, so a
    # `length` (truncated output) turn used to be indistinguishable from a
    # healthy tool call in the record.
    reported = acc.get("finish_reason")
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content,
        "input_tokens": acc["input_tokens"] or None,
        "output_tokens": acc["output_tokens"] or None,
        "finish_reason": (
            "error"
            if stream_error
            else (reported or ("tool_calls" if pending_calls else "stop"))
        ),
    }
    if stream_error:
        assistant_msg["error"] = {"message": stream_error}
    if tool_calls_meta:
        assistant_msg["tool_calls"] = tool_calls_meta
    assistant_index = len(messages)
    messages.append(assistant_msg)

    for tr in acc["tool_results"]:
        result_payload: Any = tr["content"]
        try:
            result_payload = json.loads(tr["content"])
        except (TypeError, ValueError):
            pass
        # Bound the copy that goes into history. The model already received the
        # full result during this turn; what gets stored is what every LATER
        # turn will resend, and that is where an oversized result actually
        # costs money. Propose/confirm envelopes are exempt inside prune_result.
        if prune_config is not None and prune_config.enabled:
            pruned = prune_result(
                result_payload,
                threshold_chars=prune_config.threshold_chars,
                head_chars=prune_config.head_chars,
                tail_chars=prune_config.tail_chars,
                keep_items=prune_config.keep_items,
            )
            if pruned is not None:
                logger.info(
                    "pruned tool result name=%s conv=%s",
                    tr.get("name"), conversation_id,
                )
                result_payload = pruned
        messages.append(
            {
                "role": "tool",
                "content": {
                    "tool_call_id": tr.get("tool_call_id"),
                    "name": tr.get("name"),
                    "result": result_payload,
                },
                "parent_index": assistant_index,
            }
        )

    await backend.append_messages(
        conversation_id,
        messages=messages,
        usage={"input_tokens": acc["input_tokens"], "output_tokens": acc["output_tokens"]},
        model=model,
    )
