from __future__ import annotations

from typing import Sequence

from .policies import Policy


GENERATOR_SYSTEM = (
    "You are the Generator (G) inside the CDQ curiosity optimizer. "
    "Your task is to craft concrete, research-level questions using ONLY the provided corpus snippets. "
    "Each question must reference cited evidence, highlight uncertainties, and stay grounded in the corpus. "
    "Return JSON with a list under the key 'questions'. Never output prose, code fences, or explanations."
)


def generator_prompt(
    policy: Policy,
    target_dimension: str,
    topic_summary: str,
    corpus_snippets: Sequence[str],
    num_questions: int,
) -> str:
    snippet_block = "\n".join(f"- {snippet}" for snippet in corpus_snippets)
    return (
        f"Target dimension: {target_dimension}.\n"
        f"Topic summary: {topic_summary}\n"
        f"Policy instruction: {policy.instruction}\n"
        f"Corpus snippets (ground truth context, cite them explicitly):\n{snippet_block}\n\n"
        f"Produce {num_questions} high-leverage questions that expose knowledge gaps, "
        f"disagreements, or decision-relevant uncertainties for {target_dimension}. "
        "Respond ONLY with JSON matching exactly this template:\n"
        "{\"questions\": [\"Question 1?\", \"Question 2?\", \"Question 3?\"]}\n"
        "Do NOT add numbering prefixes, markdown, or extra commentary."
    )


EVALUATOR_SYSTEM = (
    "You are the Evaluator (E). You must answer questions ONLY using the provided corpus snippets. "
    "If the snippets lack information, say Unknown. Return JSON showing 'answer' (<=80 tokens) "
    "and 'tag' as one token in {Answerable, Partial, Unknown}."
)

CLARITY_EVALUATOR_SYSTEM = (
    "You are the Clarity Evaluator. Answer the question using BOTH the provided corpus snippets "
    "and your broader knowledge. Propose a concrete research idea or experiment that leverages the "
    "Novelty/Feasibility attribute levers and explicitly map how the idea toggles them."
)


def evaluator_prompt(question: str, context_snippets: Sequence[str]) -> str:
    context_block = "\n".join(f"- {snippet}" for snippet in context_snippets)
    return (
        f"Question: {question}\n"
        f"Corpus snippets:\n{context_block}\n\n"
        "Reason strictly over this corpus. Format: {\"answer\": \"...\", \"tag\": \"Answerable\"}."
    )


def clarity_evaluator_prompt(question: str, context_snippets: Sequence[str], target_dimension: str) -> str:
    context_block = "\n".join(f"- {snippet}" for snippet in context_snippets)
    panel = ATTRIBUTE_PANELS[target_dimension]
    panel_text = "\n".join(f"{name}: {desc}" for name, desc in panel.items())
    return (
        f"Question: {question}\n"
        f"Reference snippets:\n{context_block}\n\n"
        f"Attribute levers for {target_dimension}:\n{panel_text}\n\n"
        "Design a specific research idea or experiment addressing the question (<=150 tokens). "
        "State whether the idea is grounded in the snippets, parametric knowledge, or both. "
        "List only the levers from this target dimension that the idea activates (e.g., N1,N4 for Novelty or F2,F3 for Feas). "
        "Do NOT include levers from the other dimension. Touch as many relevant levers as you reasonably can. "
        "Respond as JSON: {\"idea\": \"...\", \"grounding\": \"Corpus|Parametric|Mixed\", \"levers\": [\"N1\", \"N4\"]}."
    )


JUDGE_SYSTEM = (
    "You are the Judge (J). You issue one-token verdicts for tags, entailment, "
    "pairwise relations, and attribute panels. You never hallucinate."
)


def gap_proxy_prompt(question: str, context_snippets: Sequence[str]) -> str:
    snippets = "\n".join(f"- {s}" for s in context_snippets)
    return (
        f"Question (claim style): {question}\n"
        f"Corpus snippets:\n{snippets}\n\n"
        "Return JSON with keys 'tag' (Answerable/Partial/Unknown) and "
        "'entailment' (Entailed/Contradicted/NotEntailed)."
    )


def pairwise_prompt(answer_a: str, answer_b: str) -> str:
    return (
        "Compare the two short answers about the same corpus.\n"
        f"A: {answer_a}\nB: {answer_b}\n"
        "Output JSON {\"relation\": token} where token is one of Entails, Contradicts, Unrelated."
    )

"""
ATTRIBUTE_PANELS = {
    "Novelty": {
        "N1": "Mechanism/abstraction transfer: proposes porting/adapting a method/mechanism from another domain/task",
        "N2": "Boundary/regime/counterfactual: explores edge cases, regime shifts, stress/break conditions, or counterfactual scenarios",
        "N3": "Constructive tension: explicitly notes contradictions, gaps, or tensions with existing corpus claims/findings",
        "N4": "Evaluation/measurement innovation: introduces new metrics, benchmarks, protocols, stress tests, or data schemes",
        "N5": "Generative reach: expands the solution/idea space (new combinations, synthesis, alternative approaches/architectures)",
    },
    "Feas": {
        "F1": "Measurable outcomes: concrete success metrics/endpoints are identified",
        "F2": "Evidence/data path: data sources, collection, or evidence acquisition are specified",
        "F3": "Procedure sketch: steps/recipe/implementation plan is outlined",
        "F4": "Resource/constraint fit: compute/data/time/privacy/safety/latency constraints are addressed",
        "F5": "Validity/confound control: controls, baselines, ablations, bias checks, or threat mitigation are discussed",
    },
}
"""

ATTRIBUTE_PANELS = {
    "Novelty": {
        # CORE LEVERS (keep, slightly sharpened)
        "N1": "Mechanism/abstraction transfer: proposes adapting a mechanism, model, or method from one domain/task to a new domain/task in a non-trivial way, with some argument for why/how the transfer might work.",
        "N2": "Boundary/regime/counterfactual: explores edge cases, new regimes, failure modes, or counterfactual worlds where existing theories/methods are untested, expected to break, or give conflicting predictions.",
        "N3": "Constructive tension & theory-challenge: explicitly identifies contradictions, gaps, or competing explanations in prior work and proposes a way to resolve, unify, or decisively test them.",
        "N4": "Evaluation/measurement innovation: introduces new metrics, benchmarks, tasks, protocols, or data-collection schemes that would change how the community evaluates this phenomenon or class of methods.",
        "N5": "Generative reach & synthesis: expands the solution/idea space (e.g., new decompositions, modular combinations, or integrative frameworks) rather than proposing a single, isolated point solution.",

        # EXTENDED LEVERS (added perspectives)
        "N6": "Data/observation novelty: proposes collecting or leveraging qualitatively new kinds of data or observations (new modality, population, environment, scale, or instrument) that enable questions previously impossible or impractical.",
        "N7": "Problem framing / question novelty: formulates a genuinely new or substantially reframed research question, objective, or hypothesis rather than a small variant of standard tasks or benchmarks.",
        "N8": "Domain/context expansion: moves an established idea or tool into a substantially new scientific, application, or societal domain where it has not been systematically explored, with a plausible argument that this opens a new line of work there.",
    },
    "Feas": {
        # CORE LEVERS (keep, slightly sharpened)
        "F1": "Measurable outcomes: clearly defined success criteria, hypotheses, or evaluation metrics are stated so that progress and success can be judged objectively.",
        "F2": "Evidence/data path: there is a concrete plan for obtaining the necessary data or evidence (existing datasets, experiments, observations, or surveys), including at least a rough sense of availability or sample size.",
        "F3": "Procedure / design sketch: outlines key steps, experimental design, or algorithmic pipeline (even at a high level), making it clear how the idea would actually be executed.",
        "F4": "Resource / constraint fit: explicitly considers compute, equipment, data access, ethics/privacy/safety, and time, and argues that the project is plausible under typical lab/organization constraints.",
        "F5": "Validity / confound control: anticipates main threats to validity (confounds, biases, baselines, robustness issues) and proposes controls, comparisons, ablations, or analysis strategies to mitigate them.",

        # EXTENDED LEVERS (added perspectives)
        "F6": "Skill / expertise alignment: the required skills, infrastructure, or collaborations are realistic for a strong PhD student or typical research team; the writeup does not quietly assume capabilities that are effectively out-of-reach (e.g., 'just build GPT-7 from scratch').",
        "F7": "Timeline & decomposition: decomposes the project into stages or milestones (minimal viable experiment, main study, extensions) and is appropriately scoped for a realistic research timescale (e.g., 6–36 months).",
        "F8": "Risk management & fallback paths: identifies major technical or experimental risks and proposes fallback strategies, simplifications, or alternative paths that would still yield useful knowledge if the main plan fails.",
    },
}



def attribute_prompt(
    target_dimension: str,
    context: str,
    *,
    mode: str = "baseline",
) -> str:
    panel = ATTRIBUTE_PANELS[target_dimension]
    panel_text = "\n".join(f"{name}: {desc}" for name, desc in panel.items())
    if mode == "baseline":
        instructions = (
            "Assess only the reference corpus. Use Uncertain when the snippets do not mention a lever. "
            "Return JSON mapping each attribute key to {Yes, No, Uncertain}."
        )
    else:
        instructions = (
            "The context is decomposed into clearly labeled sections: CORPUS_SNIPPETS, QUESTION, and IDEA. "
            "Decide for each attribute whether the IDEA reduces uncertainty relative to the corpus. "
            "If the IDEA lists a lever in CLAIMED_LEVERS and you see no contradiction, default to Yes (the author is asserting it); set to No only if the idea clearly rules it out. "
            "For levers not listed, mark Yes if the IDEA contains any trigger for that lever, No if it contradicts or rules it out, "
            "and reserve Uncertain only when neither the corpus nor the idea provides evidence. Avoid marking everything Uncertain; prefer Yes/No when any cue exists. "
            "Return JSON mapping each attribute key to {Yes, No, Uncertain}."
        )
    return (
        f"Target dimension: {target_dimension}\n"
        f"Context for labeling:\n{context}\n\n"
        f"Attributes:\n{panel_text}\n"
        f"{instructions}"
    )
