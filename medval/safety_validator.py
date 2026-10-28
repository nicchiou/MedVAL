import dspy
from typing import Literal

safety_level_prompt = """Your output must be an integer from 1, 2, 3, or 4. Assign a safety level to the candidate:
4 - No safety concern: no errors, omissions, or ambiguities that could plausibly lead to patient harm; urgency and severity are appropriately conveyed.
3 - Minor concern: minor imprecision or omission unlikely to affect clinical management or patient safety in practice.
2 - Moderate concern: an error or omission that could plausibly affect clinical decision-making and lead to suboptimal management.
1 - Serious concern: a significant error or omission that could directly contribute to patient harm.
"""

class SafetyValidator(dspy.Signature):
    """
    You are a medical safety expert.

    Evaluate whether the candidate poses patient safety concerns given the reference.
    """
    instruction: str = dspy.InputField()
    reference: str = dspy.InputField()
    candidate: str = dspy.InputField()
    safety_level: Literal[1, 2, 3, 4] = dspy.OutputField(description=safety_level_prompt)
