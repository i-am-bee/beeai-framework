import os
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from eval.model import DeepEvalLLM



contextual_precision = ContextualPrecisionMetric(model=DeepEvalLLM.from_name(os.environ["EVAL_CHAT_MODEL_NAME"]))
contextual_recall = ContextualRecallMetric(model=DeepEvalLLM.from_name(os.environ["EVAL_CHAT_MODEL_NAME"]))
contextual_relevancy = ContextualRelevancyMetric(model=DeepEvalLLM.from_name(os.environ["EVAL_CHAT_MODEL_NAME"]))

test_case_1 = LLMTestCase(
    input="I'm on an F-1 visa, how long can I stay in the US after graduation?",
    actual_output="You can stay up to 30 days after completing your degree.",
    expected_output="You can stay up to 60 days after completing your degree.",
    retrieval_context=[
        """If you are in the U.S. on an F-1 visa, you are allowed to stay for 60 days after completing
        your degree, unless you have applied for and been approved to participate in OPT."""
    ]
)

test_cases = [
    LLMTestCase(
        input="Can I work off-campus on an F-1 visa without authorization?",
        actual_output="Yes, you can work off-campus without restrictions.",
        expected_output="No, you cannot work off-campus without prior authorization from USCIS.",
        retrieval_context=[
            """Students on F-1 visas are generally prohibited from off-campus employment unless they 
            have received explicit authorization from USCIS or qualify under specific training programs like OPT or CPT."""
        ]
    ),
    LLMTestCase(
        input="How many hours can I work per week on-campus with an F-1 visa?",
        actual_output="You can work 40 hours per week during the semester.",
        expected_output="You can work up to 20 hours per week during the semester.",
        retrieval_context=[
            """F-1 students are allowed to work up to 20 hours per week on-campus while school is in session. 
            They may work full-time during official school breaks."""
        ]
    ),
    LLMTestCase(
        input="Do I need a new visa if I travel abroad during OPT?",
        actual_output="You never need a new visa to reenter during OPT.",
        expected_output="You may need a valid visa stamp to reenter the U.S. during OPT.",
        retrieval_context=[
            """While on OPT, students must have a valid F-1 visa stamp in their passport to reenter the United States. 
            If the visa has expired, they must apply for a new one abroad before returning."""
        ]
    ),
    LLMTestCase(
        input="What is the maximum duration of STEM OPT extension?",
        actual_output="The STEM OPT extension lasts for 12 months.",
        expected_output="The STEM OPT extension lasts for 24 months.",
        retrieval_context=[
            """Eligible F-1 students with STEM degrees may apply for a 24-month extension of their post-completion OPT, 
            in addition to the initial 12 months."""
        ]
    ),
    LLMTestCase(
        input="Can F-1 students apply for a green card directly?",
        actual_output="Yes, F-1 students can apply for a green card directly without any restrictions.",
        expected_output="No, F-1 students cannot directly apply for a green card; they must change to an eligible status first.",
        retrieval_context=[
            """The F-1 visa is a non-immigrant visa and does not provide a direct path to permanent residency. 
            Students must transition through a work visa or family sponsorship to pursue a green card."""
        ]
    ),
    LLMTestCase(
        input="Can I transfer to another school while on an F-1 visa?",
        actual_output="No, once you choose a school, you cannot transfer.",
        expected_output="Yes, you can transfer to another SEVP-certified school if proper transfer procedures are followed.",
        retrieval_context=[
            """F-1 students may transfer between SEVP-certified schools provided they notify their current DSO, 
            maintain valid status, and complete SEVIS transfer procedures."""
        ]
    ),
    LLMTestCase(
        input="Do F-1 students have to take a full course load every semester?",
        actual_output="No, full-time enrollment is optional for F-1 students.",
        expected_output="Yes, F-1 students are required to maintain a full course of study each semester, with limited exceptions.",
        retrieval_context=[
            """Maintaining F-1 status requires full-time enrollment each semester, unless granted an authorized 
            reduced course load by the DSO for specific reasons (medical, final term, academic difficulty)."""
        ]
    ),
    LLMTestCase(
        input="How long does it take to receive an OPT approval notice?",
        actual_output="OPT approval is instant once you apply online.",
        expected_output="OPT approval typically takes 2–3 months after USCIS receives the application.",
        retrieval_context=[
            """USCIS usually processes OPT applications within 2–3 months, though times may vary. 
            Students are encouraged to apply up to 90 days before program completion."""
        ]
    ),
    LLMTestCase(
        input="Can I stay in the U.S. if my F-1 visa expires but my I-20 is still valid?",
        actual_output="No, your visa must remain valid at all times to stay in the U.S.",
        expected_output="Yes, you can stay as long as your I-20 and status are valid; the visa is only needed for reentry.",
        retrieval_context=[
            """An F-1 visa is required for entry to the United States, but not for remaining in the country. 
            Students may stay legally as long as they maintain status with a valid I-20, even if the visa expires."""
        ]
    ),
    LLMTestCase(
        input="What happens if I fall out of status on an F-1 visa?",
        actual_output="Nothing happens, you can just continue studying as usual.",
        expected_output="If you fall out of status, you may need to apply for reinstatement or leave the U.S. immediately.",
        retrieval_context=[
            """If an F-1 student falls out of status, they may be required to leave the U.S. immediately unless 
            reinstated by USCIS. Reinstatement is possible under specific conditions with proper application."""
        ]
    )
]

if __name__ == "__main__":
    eval_results = evaluate(
        test_cases=test_cases,
        metrics=[contextual_precision, contextual_recall, contextual_relevancy]
    )
    print(eval_results)