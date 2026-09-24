import asyncio
import hashlib
import json
import sys
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import JSONToolOutput, Tool, ToolRunOptions

Verdict = Literal["approve", "require_approval", "reject"]


class GovernedCustomerUpdateInput(BaseModel):
    case_id: str = Field(description="Support case identifier.")
    update_text: str = Field(description="The proposed customer-support update.")
    destination: str = Field(
        description="Where the update would be sent, such as 'internal_ticket' or 'customer_email'."
    )
    customer_impact: str = Field(description="Expected customer impact if the action is executed.")


class ActionEnvelope(BaseModel):
    tool_name: str
    action_kind: str
    proposed_action: str
    target_resource: str
    customer_impact: str


class CheckpointDecision(BaseModel):
    decision_id: str
    verdict: Verdict
    action_hash: str
    reason: str


class ActionCheckpoint(Protocol):
    async def review(self, envelope: ActionEnvelope, action_hash: str) -> CheckpointDecision: ...


def canonical_json(value: BaseModel | dict[str, object]) -> str:
    raw = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(raw, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class LocalActionCheckpoint:
    """Small in-process checkpoint used only by this example.

    Production deployments can replace this class with an HTTP call to an
    external governance service. The important pattern is to bind the decision
    to the canonical action hash before the tool performs a side effect.
    """

    async def review(self, envelope: ActionEnvelope, action_hash: str) -> CheckpointDecision:
        if envelope.target_resource != "internal_ticket":
            verdict: Verdict = "reject"
            reason = "Customer-facing destinations require a separate approval path."
        elif "refund" in envelope.proposed_action.lower():
            verdict = "require_approval"
            reason = "Refund-related updates must pause for human approval."
        else:
            verdict = "approve"
            reason = "Internal ticket update stays inside the approved operational boundary."

        decision_id = sha256_text(canonical_json({"action_hash": action_hash, "reason": reason, "verdict": verdict}))
        return CheckpointDecision(decision_id=decision_id, verdict=verdict, action_hash=action_hash, reason=reason)


class GovernedCustomerUpdateTool(Tool[GovernedCustomerUpdateInput, ToolRunOptions, JSONToolOutput[dict[str, object]]]):
    name = "GovernedCustomerUpdate"
    description = "Drafts a customer-support update only after an action-bound governance checkpoint approves it."
    input_schema = GovernedCustomerUpdateInput

    def __init__(self, checkpoint: ActionCheckpoint | None = None) -> None:
        super().__init__()
        self.checkpoint = checkpoint or LocalActionCheckpoint()

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "example", "governed_customer_update"],
            creator=self,
        )

    async def _run(
        self, input: GovernedCustomerUpdateInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput[dict[str, object]]:
        envelope = ActionEnvelope(
            tool_name=self.name,
            action_kind="support_case_update",
            proposed_action=input.update_text,
            target_resource=input.destination,
            customer_impact=input.customer_impact,
        )
        action_hash = sha256_text(canonical_json(envelope))

        decision = await self.checkpoint.review(envelope, action_hash)
        if decision.action_hash != action_hash:
            return JSONToolOutput(
                {
                    "status": "blocked",
                    "executed": False,
                    "action_hash": action_hash,
                    "decision_id": decision.decision_id,
                    "reason": "Checkpoint decision did not bind to the proposed action hash.",
                }
            )

        if decision.verdict != "approve":
            return JSONToolOutput(
                {
                    "status": decision.verdict,
                    "executed": False,
                    "action_hash": action_hash,
                    "decision_id": decision.decision_id,
                    "reason": decision.reason,
                }
            )

        executed_update = {
            "case_id": input.case_id,
            "destination": input.destination,
            "update_text": input.update_text,
        }
        return JSONToolOutput(
            {
                "status": "completed",
                "executed": True,
                "action_hash": action_hash,
                "decision_id": decision.decision_id,
                "reason": decision.reason,
                "side_effect": executed_update,
            }
        )


async def main() -> None:
    tool = GovernedCustomerUpdateTool()

    examples = [
        GovernedCustomerUpdateInput(
            case_id="CASE-1001",
            update_text="Add an internal note that the package was reshipped.",
            destination="internal_ticket",
            customer_impact="No direct customer-facing message.",
        ),
        GovernedCustomerUpdateInput(
            case_id="CASE-1002",
            update_text="Add an internal note recommending a refund review.",
            destination="internal_ticket",
            customer_impact="May influence refund handling.",
        ),
        GovernedCustomerUpdateInput(
            case_id="CASE-1003",
            update_text="Send the customer a final resolution message.",
            destination="customer_email",
            customer_impact="Customer-facing communication.",
        ),
    ]

    for example in examples:
        result = await tool.run(example)
        print(result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        sys.exit(e.explain())
