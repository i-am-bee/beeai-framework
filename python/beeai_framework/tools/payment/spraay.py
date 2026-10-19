# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""Spraay batch payment tool (author: Plagtech).

Lets agents validate, estimate and prepare USDC batch payments on Base through the Spraay gateway
(https://docs.spraay.app). The tool is non-custodial: it never touches private keys and never signs anything.
The ``execute`` action returns unsigned transactions that the sender signs with their own wallet.
"""

import re
from collections.abc import Sequence
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any, Literal, Self

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.tools import JSONToolOutput, ToolError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions

logger = Logger(__name__)

SPRAAY_GATEWAY_URL = "https://gateway.spraay.app"
SPRAAY_BATCH_CONTRACT = "0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC"
SPRAAY_CHAIN = "base"
SPRAAY_CHAIN_ID = 8453
SPRAAY_MAX_RECIPIENTS = 200
SPRAAY_PROTOCOL_FEE_BPS = 30
SPRAAY_TOKEN_DECIMALS: dict[str, int] = {"USDC": 6}

SpraayAction = Literal["get_prices", "validate_batch", "estimate_batch", "execute_batch"]

_ADDRESS_PATTERN = re.compile(r"^0x[0-9a-fA-F]{40}$")
_BATCH_ACTIONS: frozenset[str] = frozenset({"validate_batch", "estimate_batch", "execute_batch"})


def to_base_units(amount: str, decimals: int) -> int:
    """Convert a human-readable token amount (e.g. "1.5") to raw base units (e.g. 1500000 for 6 decimals)."""
    try:
        value = Decimal(amount.strip())
    except InvalidOperation as e:
        raise ValueError(f"'{amount}' is not a valid number.") from e

    if not value.is_finite() or value <= 0:
        raise ValueError(f"'{amount}' must be a positive number.")

    raw = value.scaleb(decimals)
    if raw != raw.to_integral_value():
        raise ValueError(f"'{amount}' has more than {decimals} decimal places.")
    return int(raw)


def from_base_units(raw: int, decimals: int) -> str:
    """Convert raw base units back to a human-readable decimal string."""
    return format(Decimal(raw).scaleb(-decimals).normalize(), "f")


class SpraayBatchPaymentToolInput(BaseModel):
    action: SpraayAction = Field(
        description=(
            "Operation to perform: 'get_prices' (current token price), 'validate_batch' (check recipients and "
            "amounts), 'estimate_batch' (gas and protocol fee estimate) or 'execute_batch' (build the unsigned "
            "batch payment transaction for the sender to sign)."
        )
    )
    recipients: list[str] = Field(
        default_factory=list,
        max_length=SPRAAY_MAX_RECIPIENTS,
        description=(
            f"Recipient wallet addresses on Base (0x followed by 40 hex characters). "
            f"At most {SPRAAY_MAX_RECIPIENTS}. Required for every action except 'get_prices'."
        ),
    )
    amounts: list[str] = Field(
        default_factory=list,
        max_length=SPRAAY_MAX_RECIPIENTS,
        description=(
            "Human-readable token amounts, one per recipient in the same order, e.g. ['1.5', '20']. "
            "Do NOT convert to base units; the tool does that. Required for every action except 'get_prices'."
        ),
    )
    token: Literal["USDC"] = Field(default="USDC", description="Token to send.")
    sender: str | None = Field(
        default=None,
        description="Wallet address that will sign and fund the batch. Required for 'execute_batch'.",
    )

    @field_validator("amounts", mode="before")
    @classmethod
    def _amounts_to_str(cls, value: Any) -> Any:
        if isinstance(value, Sequence) and not isinstance(value, str):
            return [str(v) if isinstance(v, int | float | Decimal) and not isinstance(v, bool) else v for v in value]
        return value

    @field_validator("recipients")
    @classmethod
    def _validate_recipients(cls, value: list[str]) -> list[str]:
        value = [address.strip() for address in value]
        invalid = [address for address in value if not _ADDRESS_PATTERN.match(address)]
        if invalid:
            raise ValueError(f"Invalid address(es), expected 0x followed by 40 hex characters: {', '.join(invalid)}")
        return value

    @field_validator("sender")
    @classmethod
    def _validate_sender(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not _ADDRESS_PATTERN.match(value):
            raise ValueError(f"Invalid sender address '{value}', expected 0x followed by 40 hex characters.")
        return value

    @model_validator(mode="after")
    def _validate_batch(self) -> Self:
        if self.action not in _BATCH_ACTIONS:
            return self

        if not self.recipients:
            raise ValueError(f"'recipients' must not be empty for action '{self.action}'.")
        if len(self.recipients) != len(self.amounts):
            raise ValueError(
                f"'recipients' and 'amounts' must have the same length "
                f"(got {len(self.recipients)} recipients and {len(self.amounts)} amounts)."
            )
        decimals = SPRAAY_TOKEN_DECIMALS[self.token]
        for index, amount in enumerate(self.amounts):
            try:
                to_base_units(amount, decimals)
            except ValueError as e:
                raise ValueError(f"amounts[{index}]: {e}") from e
        if self.action == "execute_batch" and self.sender is None:
            raise ValueError("'sender' is required for action 'execute_batch'.")
        return self

    def raw_amounts(self) -> list[int]:
        decimals = SPRAAY_TOKEN_DECIMALS[self.token]
        return [to_base_units(amount, decimals) for amount in self.amounts]


class SpraayBatchPaymentToolOutput(JSONToolOutput[dict[str, Any]]):
    pass


class SpraayBatchPaymentTool(Tool[SpraayBatchPaymentToolInput, ToolRunOptions, SpraayBatchPaymentToolOutput]):
    name = "SpraayBatchPayment"
    description = (
        f"Pay many recipients in one on-chain USDC batch transaction on Base via the Spraay Protocol "
        f"(up to {SPRAAY_MAX_RECIPIENTS} recipients, {SPRAAY_PROTOCOL_FEE_BPS / 100}% protocol fee). "
        "Always 'validate_batch' and 'estimate_batch' before 'execute_batch'. Amounts are human-readable "
        "(e.g. '1.5' USDC). Non-custodial: 'execute_batch' returns unsigned transactions that the sender "
        "must sign with their own wallet; this tool never signs or sends funds."
    )
    input_schema = SpraayBatchPaymentToolInput

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        base_url: str = SPRAAY_GATEWAY_URL,
        http_client: httpx.AsyncClient | None = None,
        allowed_actions: Sequence[SpraayAction] | None = None,
        timeout: float = 30,
    ) -> None:
        """Create the tool.

        Args:
            options: Generic tool options (e.g. cache).
            base_url: Spraay gateway URL.
            http_client: Optional pre-configured client. ``execute_batch`` is an x402-paid endpoint; pass a
                client that settles x402 payments (e.g. one wrapped by an x402 SDK) to execute batches.
                Without one, ``execute_batch`` reports the payment requirements instead of failing.
            allowed_actions: Restrict which actions this instance may perform, e.g. give a research agent
                only ``["get_prices", "validate_batch", "estimate_batch"]``. All actions are allowed by default.
            timeout: HTTP timeout in seconds.
        """
        super().__init__(options)
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client
        self.allowed_actions: tuple[SpraayAction, ...] | None = (
            tuple(allowed_actions) if allowed_actions is not None else None
        )
        self.timeout = timeout

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "payment", "spraay"],
            creator=self,
        )

    async def clone(self) -> Self:
        tool = self.__class__(
            options=self.options,
            base_url=self.base_url,
            http_client=self.http_client,
            allowed_actions=self.allowed_actions,
            timeout=self.timeout,
        )
        tool.name = self.name
        tool.description = self.description
        tool.input_schema = self.input_schema
        tool.middlewares.extend(self.middlewares)
        tool._cache = await self.cache.clone()
        return tool

    async def _run(
        self, input: SpraayBatchPaymentToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> SpraayBatchPaymentToolOutput:
        if self.allowed_actions is not None and input.action not in self.allowed_actions:
            raise ToolError(
                f"Action '{input.action}' is not allowed for this tool instance. "
                f"Allowed actions: {', '.join(self.allowed_actions)}."
            )

        if input.action == "get_prices":
            result = await self._get_prices(input)
        elif input.action == "validate_batch":
            result = await self._validate_batch(input)
        elif input.action == "estimate_batch":
            result = await self._estimate_batch(input)
        else:
            result = await self._execute_batch(input)

        return SpraayBatchPaymentToolOutput(result)

    async def _get_prices(self, input: SpraayBatchPaymentToolInput) -> dict[str, Any]:
        response = await self._request("GET", "/free/prices", params={"token": input.token})
        payload = self._parse_json(response)
        return {"action": input.action, "token": input.token, "prices": payload.get("prices", {})}

    async def _validate_batch(self, input: SpraayBatchPaymentToolInput) -> dict[str, Any]:
        body = {
            "chain": SPRAAY_CHAIN,
            "token": input.token,
            "recipients": [
                {"to": to, "amount": str(amount)}
                for to, amount in zip(input.recipients, input.raw_amounts(), strict=True)
            ],
        }
        if input.sender:
            body["sender"] = input.sender

        response = await self._request("POST", "/free/validate-batch", json=body)
        payload = self._parse_json(response)
        return {
            "action": input.action,
            "valid": payload.get("valid", False),
            "errors": payload.get("errors", []),
            "warnings": payload.get("warnings", []),
            "batch": self._summarize(input),
        }

    async def _estimate_batch(self, input: SpraayBatchPaymentToolInput) -> dict[str, Any]:
        summary = self._summarize(input)
        response = await self._request(
            "GET",
            "/free/estimate-batch",
            params={
                "chain": SPRAAY_CHAIN,
                "token": input.token,
                "recipients": len(input.recipients),
                "amount": summary["total_amount"],
            },
        )
        payload = self._parse_json(response)
        return {"action": input.action, "estimate": payload.get("estimate", {}), "batch": summary}

    async def _execute_batch(self, input: SpraayBatchPaymentToolInput) -> dict[str, Any]:
        body = {
            "token": input.token,
            "sender": input.sender,
            "recipients": input.recipients,
            "amounts": [str(amount) for amount in input.raw_amounts()],
        }
        response = await self._request("POST", "/api/v1/batch/execute", json=body)
        summary = self._summarize(input)

        if response.status_code == httpx.codes.PAYMENT_REQUIRED:
            return {
                "action": input.action,
                "status": "payment_required",
                "message": (
                    "The Spraay execute endpoint requires an x402 micropayment. Configure the tool with an "
                    "x402-enabled http_client to build the unsigned batch transaction."
                ),
                "payment_requirements": self._parse_json(response, allow_status=True).get("accepts", []),
                "batch": summary,
            }

        payload = self._parse_json(response)
        return {
            "action": input.action,
            "status": "unsigned_transaction_ready",
            "message": "Sign and broadcast these transactions with the sender wallet. Nothing has been sent yet.",
            "transactions": payload.get("transactions", []),
            "batch": summary,
            "gateway_response": payload,
        }

    def _summarize(self, input: SpraayBatchPaymentToolInput) -> dict[str, Any]:
        decimals = SPRAAY_TOKEN_DECIMALS[input.token]
        total_raw = sum(input.raw_amounts())
        fee_raw = int((Decimal(total_raw) * SPRAAY_PROTOCOL_FEE_BPS / 10_000).to_integral_value(ROUND_CEILING))
        return {
            "chain": SPRAAY_CHAIN,
            "chain_id": SPRAAY_CHAIN_ID,
            "contract": SPRAAY_BATCH_CONTRACT,
            "token": input.token,
            "sender": input.sender,
            "recipient_count": len(input.recipients),
            "total_amount": from_base_units(total_raw, decimals),
            "total_amount_base_units": str(total_raw),
            "protocol_fee_bps": SPRAAY_PROTOCOL_FEE_BPS,
            "protocol_fee": from_base_units(fee_raw, decimals),
            "total_with_fee": from_base_units(total_raw + fee_raw, decimals),
        }

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self.base_url}{path}"
        logger.debug(f"Spraay request: {method} {url}")
        try:
            if self.http_client is not None:
                return await self.http_client.request(method, url, timeout=self.timeout, **kwargs)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return await client.request(method, url, **kwargs)
        except httpx.HTTPError as e:
            raise ToolError(f"Could not reach the Spraay gateway at {self.base_url}.", cause=e) from e

    @staticmethod
    def _parse_json(response: httpx.Response, *, allow_status: bool = False) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if not allow_status and response.is_error:
            detail = ""
            if isinstance(payload, dict):
                detail = str(payload.get("error") or payload.get("message") or payload.get("errors") or "")
            raise ToolError(
                f"Spraay gateway returned HTTP {response.status_code} for {response.request.url.path}"
                + (f": {detail}" if detail else "."),
                context={"status_code": response.status_code},
            )

        if not isinstance(payload, dict):
            raise ToolError("Spraay gateway returned an unexpected (non-JSON object) response.")
        return payload
