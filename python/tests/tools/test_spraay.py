# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable

import httpx
import pytest

from beeai_framework.tools import ToolError, ToolInputValidationError
from beeai_framework.tools.payment import SpraayBatchPaymentTool, SpraayBatchPaymentToolOutput
from beeai_framework.tools.payment.spraay import SpraayAction, from_base_units, to_base_units

SENDER = "0x1111111111111111111111111111111111111111"
ALICE = "0x2222222222222222222222222222222222222222"
BOB = "0x3333333333333333333333333333333333333333"


def make_tool(
    handler: Callable[[httpx.Request], httpx.Response], allowed_actions: list[SpraayAction] | None = None
) -> SpraayBatchPaymentTool:
    return SpraayBatchPaymentTool(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), allowed_actions=allowed_actions
    )


def no_request(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"Unexpected request to {request.url}")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("amount", "expected"),
    [("1", 1_000_000), ("1.5", 1_500_000), ("0.000001", 1), ("20.25", 20_250_000), (" 3 ", 3_000_000)],
)
def test_to_base_units(amount: str, expected: int) -> None:
    assert to_base_units(amount, 6) == expected


@pytest.mark.unit
@pytest.mark.parametrize("amount", ["0", "-1", "abc", "1.0000001", "NaN", "Infinity"])
def test_to_base_units_rejects(amount: str) -> None:
    with pytest.raises(ValueError):
        to_base_units(amount, 6)


@pytest.mark.unit
def test_from_base_units() -> None:
    assert from_base_units(1_500_000, 6) == "1.5"
    assert from_base_units(20_000_000, 6) == "20"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input",
    [
        {"action": "validate_batch", "recipients": ["0x123"], "amounts": ["1"]},
        {"action": "validate_batch", "recipients": [ALICE, BOB], "amounts": ["1"]},
        {"action": "validate_batch", "recipients": [ALICE], "amounts": ["1.1234567"]},
        {"action": "validate_batch", "recipients": [], "amounts": []},
        {"action": "execute_batch", "recipients": [ALICE], "amounts": ["1"]},
        {"action": "execute_batch", "recipients": [ALICE], "amounts": ["1"], "sender": "not-an-address"},
        {"action": "validate_batch", "recipients": [ALICE] * 201, "amounts": ["1"] * 201},
        {"action": "transfer", "recipients": [ALICE], "amounts": ["1"]},
    ],
)
async def test_invalid_input(input: dict[str, object]) -> None:
    with pytest.raises(ToolInputValidationError):
        await make_tool(no_request).run(input)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_prices() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/free/prices"
        assert request.url.params["token"] == "USDC"
        return httpx.Response(200, json={"prices": {"USDC": {"usd": 0.9998}}, "timestamp": 1})

    result = await make_tool(handler).run({"action": "get_prices"})

    assert type(result) is SpraayBatchPaymentToolOutput
    assert result.result["prices"] == {"USDC": {"usd": 0.9998}}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_batch_converts_amounts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/free/validate-batch"
        assert json.loads(request.content) == {
            "chain": "base",
            "token": "USDC",
            "recipients": [{"to": ALICE, "amount": "1500000"}, {"to": BOB, "amount": "20000000"}],
        }
        return httpx.Response(200, json={"valid": True, "errors": [], "warnings": []})

    result = await make_tool(handler).run(
        {"action": "validate_batch", "recipients": [ALICE, BOB], "amounts": ["1.5", 20]}
    )

    assert result.result["valid"] is True
    assert result.result["batch"]["total_amount"] == "21.5"
    assert result.result["batch"]["total_amount_base_units"] == "21500000"
    assert result.result["batch"]["protocol_fee"] == "0.0645"
    assert result.result["batch"]["total_with_fee"] == "21.5645"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_estimate_batch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/free/estimate-batch"
        assert dict(request.url.params) == {"chain": "base", "token": "USDC", "recipients": "2", "amount": "21.5"}
        return httpx.Response(200, json={"estimate": {"estimatedGasUSD": 0.046, "supported": True}})

    result = await make_tool(handler).run(
        {"action": "estimate_batch", "recipients": [ALICE, BOB], "amounts": ["1.5", "20"]}
    )

    assert result.result["estimate"] == {"estimatedGasUSD": 0.046, "supported": True}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_batch_returns_unsigned_transactions() -> None:
    transactions = [{"to": "0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC", "data": "0xabcdef", "value": "0"}]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/batch/execute"
        assert json.loads(request.content) == {
            "token": "USDC",
            "sender": SENDER,
            "recipients": [ALICE, BOB],
            "amounts": ["1000000", "2000000"],
        }
        return httpx.Response(200, json={"transactions": transactions})

    result = await make_tool(handler).run(
        {"action": "execute_batch", "recipients": [ALICE, BOB], "amounts": ["1", "2"], "sender": SENDER}
    )

    assert result.result["status"] == "unsigned_transaction_ready"
    assert result.result["transactions"] == transactions


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_batch_payment_required() -> None:
    accepts = [{"scheme": "exact", "network": "eip155:8453", "amount": "20000"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"x402Version": 2, "error": "Payment required", "accepts": accepts})

    result = await make_tool(handler).run(
        {"action": "execute_batch", "recipients": [ALICE], "amounts": ["1"], "sender": SENDER}
    )

    assert result.result["status"] == "payment_required"
    assert result.result["payment_requirements"] == accepts


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gateway_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "recipients must be between 1 and 200"})

    with pytest.raises(ToolError, match=r"HTTP 400.*recipients must be between 1 and 200"):
        await make_tool(handler).run({"action": "estimate_batch", "recipients": [ALICE], "amounts": ["1"]})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ToolError, match="Could not reach the Spraay gateway"):
        await make_tool(handler).run({"action": "get_prices"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_allowed_actions() -> None:
    tool = make_tool(no_request, allowed_actions=["get_prices", "validate_batch", "estimate_batch"])

    with pytest.raises(ToolError, match="not allowed"):
        await tool.run({"action": "execute_batch", "recipients": [ALICE], "amounts": ["1"], "sender": SENDER})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone() -> None:
    tool = make_tool(no_request, allowed_actions=["get_prices"])
    cloned = await tool.clone()

    assert cloned is not tool
    assert cloned.allowed_actions == ("get_prices",)
    assert cloned.http_client is tool.http_client
    assert cloned.base_url == tool.base_url
