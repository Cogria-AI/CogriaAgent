"""CogriaAgent HTTP contract: shared types, JSON Schemas, conformance suite."""

from __future__ import annotations

from typing import Any

from .types import (
    FORBIDDEN,
    PROPOSAL_EXPIRED,
    PROPOSAL_REQUIRED,
    VALIDATION_FAILED,
    Catalog,
    CatalogAction,
    Envelope,
    ErrorObj,
    ExchangeResponse,
    ProposeData,
    url_slug_for,
)

__all__ = [
    "Envelope",
    "ErrorObj",
    "ProposeData",
    "CatalogAction",
    "Catalog",
    "ExchangeResponse",
    "url_slug_for",
    "PROPOSAL_REQUIRED",
    "PROPOSAL_EXPIRED",
    "VALIDATION_FAILED",
    "FORBIDDEN",
    "json_schemas",
]


def json_schemas() -> dict[str, Any]:
    """All contract JSON Schemas, keyed by name — for docs / cross-language codegen."""
    return {
        "Envelope": Envelope.model_json_schema(),
        "CatalogAction": CatalogAction.model_json_schema(),
        "Catalog": Catalog.model_json_schema(),
        "ProposeData": ProposeData.model_json_schema(),
        "ExchangeResponse": ExchangeResponse.model_json_schema(),
    }
