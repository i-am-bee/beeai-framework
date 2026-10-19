import asyncio
import sys
import traceback
from typing import Literal

from pydantic import BaseModel, Field

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.payment import SpraayBatchPaymentTool
from beeai_framework.tools.think import ThinkTool

# The wallet that will sign the batch. The tool never sees a private key, it only builds unsigned transactions.
SENDER = "0x1111111111111111111111111111111111111111"
PAYOUTS = {
    "0x2222222222222222222222222222222222222222": "150",
    "0x3333333333333333333333333333333333333333": "75.5",
    "0x4444444444444444444444444444444444444444": "42.25",
}
BUDGET_USDC = 300


class PaymentReview(BaseModel):
    decision: Literal["APPROVED", "REJECTED"] = Field(description="Whether the batch payment may be executed.")
    reason: str = Field(description="Short justification for the decision.")


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite4:micro")

    # Agent 1: validates and estimates the batch. It cannot execute payments.
    analyst = RequirementAgent(
        llm=llm,
        tools=[SpraayBatchPaymentTool(allowed_actions=["get_prices", "validate_batch", "estimate_batch"])],
        requirements=[ConditionalRequirement(SpraayBatchPaymentTool, min_invocations=2)],
        role="Payment Analyst",
        instructions=(
            "Validate the batch with 'validate_batch', then estimate it with 'estimate_batch'. "
            "Report validity, any errors or warnings, total amount, protocol fee, total with fee and estimated gas."
        ),
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
    )

    batch = "\n".join(f"- {address}: {amount} USDC" for address, amount in PAYOUTS.items())
    analysis = await analyst.run(f"Prepare this USDC batch payment from {SENDER} on Base:\n{batch}")
    print("Analyst:", analysis.last_message.text)

    # Agent 2: reviews the analyst's report against the budget and approves or rejects it.
    reviewer = RequirementAgent(
        llm=llm,
        tools=[ThinkTool()],
        requirements=[ConditionalRequirement(ThinkTool, force_at_step=1)],
        role="Payment Approver",
        instructions=(
            f"Approve only if the batch is valid and the total including fees is at most {BUDGET_USDC} USDC. "
            "Otherwise reject it."
        ),
    )
    review = await reviewer.run(
        f"Review this batch payment report:\n{analysis.last_message.text}", expected_output=PaymentReview
    )
    decision = review.output_structured
    assert isinstance(decision, PaymentReview)
    print(f"Reviewer: {decision.decision} - {decision.reason}")

    if decision.decision != "APPROVED":
        return

    # Execution is a deterministic step gated by the approval, not an LLM decision.
    # Pass an x402-enabled `http_client` to the tool to pay the gateway fee and receive the unsigned transactions.
    executor = SpraayBatchPaymentTool(allowed_actions=["execute_batch"])
    result = await executor.run(
        {
            "action": "execute_batch",
            "sender": SENDER,
            "recipients": list(PAYOUTS.keys()),
            "amounts": list(PAYOUTS.values()),
        }
    )
    print("Execution:", result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
