"""AgentAction — base class for every business action.

Read actions override `run()` and return data. Write actions set
`requires_confirm = True` and implement the write hooks; the base runs the
propose/confirm + auto-audit orchestration. The orchestrator strips
`proposal_token` from params before calling hooks, so hooks see clean business
params. Single-tenant: hooks take (params, ctx) — no tenant.
"""

from __future__ import annotations

import uuid
from typing import Any

from .audit import AuditSink, NullAuditSink
from .context import AgentContext
from .envelope import fail, ok, propose
from .proposal import ProposalTokenService

VALIDATION_FAILED = "VALIDATION_FAILED"
PROPOSAL_REQUIRED = "PROPOSAL_REQUIRED"
PROPOSAL_EXPIRED = "PROPOSAL_EXPIRED"


class AgentAction:
    name: str = ""
    description: str = ""
    requires_confirm: bool = False
    ability: str | None = None
    target_type: str = ""

    # --- catalog metadata ----------------------------------------------------
    def params_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    def returns_schema(self) -> dict[str, Any] | None:
        return None

    def url_slug(self) -> str:
        return self.name.replace("_", "-")

    def catalog_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "name": self.name,
            "url_slug": self.url_slug(),
            "description": self.description,
            "params_schema": self.params_schema(),
            "requires_confirm": self.requires_confirm,
        }
        if self.returns_schema() is not None:
            entry["returns_schema"] = self.returns_schema()
        if self.ability:
            entry["ability"] = self.ability
        return entry

    # --- read action ---------------------------------------------------------
    async def run(self, params: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
        raise NotImplementedError(f"Read action {type(self).__name__} must implement run().")

    def read_message(self, data: dict[str, Any], params: dict[str, Any]) -> str | None:
        return None

    # --- write hooks (overridden by write subclasses) ------------------------
    def rules(self, params: dict[str, Any]) -> str | None:
        """Return an error message if params are invalid, else None. Base does a
        minimal required-fields check from params_schema."""
        required = set(self.params_schema().get("required") or [])
        missing = [k for k in required if params.get(k) in (None, "")]
        if missing:
            return f"Missing required field(s): {', '.join(missing)}"
        return None

    async def validate_target(self, params: dict[str, Any], ctx: AgentContext) -> dict[str, Any] | None:
        """Pre-flight business validation (target exists / editable). Return a
        failure envelope to refuse cleanly (no token, no audit), or None."""
        return None

    async def handle(self, params: dict[str, Any], ctx: AgentContext) -> dict[str, Any]:
        raise NotImplementedError(f"Write action {type(self).__name__} must implement handle().")

    async def capture_before(self, params: dict[str, Any], ctx: AgentContext) -> dict[str, Any] | None:
        return None

    async def capture_after(self, params: dict[str, Any], ctx: AgentContext, result: dict[str, Any]) -> dict[str, Any] | None:
        return None

    def summarize(self, before: dict | None, after: dict | None, params: dict[str, Any]) -> str:
        return self.name

    def proposal_summary(self, params: dict[str, Any]) -> str:
        return self.name

    def success_message(self, result: dict[str, Any], params: dict[str, Any]) -> str:
        return self.summarize(None, None, params)

    def extract_target_id(self, result: dict[str, Any]) -> str | None:
        return str(result["id"]) if isinstance(result, dict) and "id" in result else None

    # --- orchestration -------------------------------------------------------
    async def execute(
        self,
        *,
        params: dict[str, Any],
        ctx: AgentContext,
        dry_run: bool,
        proposal_token: str | None,
        tokens: ProposalTokenService,
        audit: AuditSink | None = None,
    ) -> dict[str, Any]:
        if not self.requires_confirm:
            data = await self.run(params, ctx)
            return ok(data, self.read_message(data, params))
        return await self._run_write(
            params=params, ctx=ctx, dry_run=dry_run, proposal_token=proposal_token, tokens=tokens, audit=audit or NullAuditSink()
        )

    async def _run_write(
        self,
        *,
        params: dict[str, Any],
        ctx: AgentContext,
        dry_run: bool,
        proposal_token: str | None,
        tokens: ProposalTokenService,
        audit: AuditSink,
    ) -> dict[str, Any]:
        err = self.rules(params)
        if err:
            return fail(VALIDATION_FAILED, err)

        target_error = await self.validate_target(params, ctx)
        if target_error is not None:
            return target_error

        # --- propose -----------------------------------------------------
        if dry_run:
            token = await tokens.issue(subject=ctx.user_id, action=self.name, params=params)
            return propose(token, self.proposal_summary(params))

        # --- confirm -----------------------------------------------------
        if not proposal_token:
            return fail(PROPOSAL_REQUIRED, "Confirmation required before this action can run.")

        payload = await tokens.consume(proposal_token, subject=ctx.user_id, action=self.name, params=params)
        if payload is None:
            return fail(PROPOSAL_EXPIRED, "This confirmation has expired or is invalid. Please try again.")

        audit_id = "act_" + uuid.uuid4().hex
        before = await self.capture_before(params, ctx)
        try:
            result = await self.handle(params, ctx)
            after = await self.capture_after(params, ctx, result)
            await audit.write(
                {
                    "audit_id": audit_id,
                    "action_name": self.name,
                    "target_type": self.target_type,
                    "target_id": self.extract_target_id(result),
                    "user_id": ctx.user_id,
                    "before_state": before,
                    "after_state": after,
                    "diff_summary": self.summarize(before, after, params),
                    "status": "success",
                }
            )
        except Exception as e:  # noqa: BLE001 — record the failure trail, then re-raise
            await audit.write(
                {
                    "audit_id": audit_id,
                    "action_name": self.name,
                    "target_type": self.target_type,
                    "user_id": ctx.user_id,
                    "before_state": before,
                    "after_state": None,
                    "diff_summary": self.summarize(before, None, params),
                    "status": "failed",
                    "error_detail": {"class": type(e).__name__, "message": str(e)},
                }
            )
            raise

        return ok({**result, "audit_id": audit_id}, self.success_message(result, params), audit_id=audit_id)
