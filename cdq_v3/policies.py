from dataclasses import dataclass
from typing import List

@dataclass
class Policy:
    name: str
    instruction: str

def default_policies() -> List[Policy]:
    """
    Pi_0 (minimal, non-overlapping).
    Each instruction explicitly uses: "Ask questions about ... to explore ...".
    """
    return [
        # 1) Knowledge-state discrepancies (unknowns & contradictions)
        Policy(
            name="info-gap-hunter",
            instruction=(
                "POLICY: Identify concrete knowledge gaps or contradictions "
                "(missing evidence, unresolved claims, incompatible findings). "
                "Ask questions about what is unknown or inconsistent to explore why these gaps arise, "
                "what evidence would decisively resolve them, and how resolution would change interpretation or practice."
            )
        ),

        # 2) Cross-domain structural transfer (distinct from counterfactuals)
        Policy(
            name="analogy-bridger",
            instruction=(
                "POLICY: Map a structurally similar problem from another field (source->target). "
                "Ask questions about the transferred mechanism to explore whether, when, and why it should hold, "
                "where it should fail, and what observations would distinguish successful transfer from failure."
            )
        ),

        # 3) Theory/assumption stress via counterfactuals & boundaries (within-domain)
        Policy(
            name="counterfactual-boundary",
            instruction=(
                "POLICY: Alter a key mechanism, assumption, or regime (invert, remove, or push to an extreme). "
                "Ask questions about the resulting consequences to explore which predictions diverge, "
                "how to discriminate between the original and altered accounts, and which interventions or observations reveal the difference."
            )
        ),

        # 4) Method/system breakdowns (distinct from knowledge contradictions)
        Policy(
            name="failure-to-question",
            instruction=(
                "POLICY: Start from an observed failure mode or limitation (e.g., optimization instability, data shift, "
                "brittleness, safety, calibration). "
                "Ask questions about likely root causes to explore minimal isolating interventions and feasible mitigations."
            )
        ),

        # 5) Constraints + evaluation
        Policy(
            name="constraint-eval-reframer",
            instruction=(
                "POLICY: Reframe the problem under real constraints (compute, data, latency, privacy, safety) "
                "and the need for sound evaluation (proxies, constructs, stress tests). "
                "Ask questions about key trade-offs and measurement choices to explore robust designs under limits, "
                "construct validity, sensitivity/specificity to the intended phenomenon, and predictive utility for downstream outcomes."
            )
        ),
    ]
