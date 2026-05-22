def get_idea_generation_prompt_template_single():
    """
    Prompt template for generating a research idea using only paper abstracts.
    """
    prompt = """You are a top-tier researcher. You are tasked with creating a research idea given some background knowledge derived from existing papers.

Background knowledge:
{title_abstract_pair}

Using the background knowledge provided, reason over the key themes, limitations, and open gaps implied by the existing work, 
and generate a research idea that demonstrates strong {quality_indicator}. 
The idea should be well-motivated by the background knowledge and reflect a coherent research direction.

Please avoid copying ideas directly; instead, synthesize and extend the insights to inspire a new research idea. 
Return your idea in the following format:

Title: [A brief, focused title]
Problem: [The core issue or gap being addressed]
Objective: [The main goal or research question]
Hypothesis: [The hypothesis being tested or explored]
Method: [The approach or methodology]
Expected Impact/Findings: [The anticipated outcomes or contributions]

Please only respond with the research idea in the format provided above. Do not respond with anything else."""
    return prompt

def get_idea_generation_prompt_template_single_questions():
    """
    Prompt template for generating a research idea using titles, abstracts, and guiding questions.
    """
    prompt = """You are a top-tier researcher. You are tasked with creating a research idea given some background knowledge and guiding questions.

The background knowledge comes from existing papers. 
The guiding questions clarify the problem setting, underlying assumptions, and practical considerations that shape the research direction.

Background knowledge:
{title_abstract_pair}

Guiding questions:
{questions}

Using both the background knowledge and the guiding questions, reason over key themes, limitations, and open gaps, 
and generate a research idea that demonstrates strong {quality_indicator}. 
The idea should be well-grounded in prior work and coherent with the problem setting implied by the guiding questions.

Please avoid copying ideas directly; instead, synthesize insights from both sources to inspire a new research idea. 
Return your idea in the following format:

Title: [A brief, focused title]
Problem: [The core issue or gap being addressed]
Objective: [The main goal or research question]
Hypothesis: [The hypothesis being tested or explored]
Method: [The approach or methodology]
Expected Impact/Findings: [The anticipated outcomes or contributions]

Please only respond with the research idea in the format provided above. Do not respond with anything else."""
    return prompt



def get_idea_generation_prompt_template_rag():
    """
    Prompt template for generating a research idea using paper abstracts and RAG-generated context.
    """
    prompt = """You are a top-tier researcher. You are tasked with creating a research idea given some background knowledge. 
        The background knowledge comes from existing papers, and the questions highlight the research directions:

        1. Background knowledge:
        {title_abstract_pair}

        2. Additional context retrieved from web search:
        {rag_context}

        Using the knowledge provided, reason over them and generate a research idea that demonstrates strong {quality_indicator}. Please avoid copying ideas directly; rather, use the insights to inspire a new research idea. Return your idea in the following format:

        Title: [A brief, focused title]
        Problem: [The core issue or gap being addressed]
        Objective: [The main goal or research question]
        Hypothesis: [The hypothesis being tested or explored]
        Method: [The approach or methodology]
        Expected Impact/Findings: [The anticipated outcomes or contributions]

        Please only respond with the research idea in the format provided above. Do not respond with anything else."""
    return prompt

# def review_prompt():
    # """
    # Prompt template for generating a research idea using only paper abstracts.
    # """
    # prompt = """You will receive the proposer's research idea. Try to give the best constructive criticism on the research idea's {quality_indicator} so that the proposer can improve the idea's {quality_indicator} as much as possible. In your response, please explain why the research idea lacks in {quality_indicator}. Here is the proposer's research idea: {research_idea}"""
    # return prompt
# 
# def review_prompt_q():
    # """
    # Prompt template for generating constructive criticism of a research idea,
    # considering both the question and the proposed idea.
    # """
    # prompt = """You are a reviewer evaluating a research proposal.  
    # Your task is to give the best constructive criticism on the research idea's {quality_indicator} with respect to some questions, 
    # so that the proposer can improve it as much as possible.  
# 
    # You must base your critique on **both** the given question and the proposed research idea.  
# 
    # **Questions:**  
    # {question}  
# 
    # **Proposed Research Idea:**  
    # {research_idea}  
# 
    # In your response, please explain clearly:  
    # 1. Why the research idea lacks in {quality_indicator} given the context of the questions.  
    # 2. How the proposer could improve the {quality_indicator} in a concrete and actionable way with respect to the questions."""
    # return prompt

def review_prompt():
    """
    Prompt template for generating constructive criticism of a research idea
    based only on the idea itself.
    """
    prompt = """You are a reviewer evaluating a research proposal.

Your task is to provide constructive criticism on the research idea's {quality_indicator}, 
helping the proposer improve it as much as possible.

Base your critique strictly on the proposed research idea itself. 
Use the idea to infer the intended problem setting, underlying assumptions, 
and evaluation context implied by the proposal.

Assess whether the idea is internally coherent and sufficiently developed under its implied setting, 
and identify gaps, ambiguities, or unrealistic assumptions that limit the {quality_indicator}. 
Explain why these issues matter given what the idea claims to address.

In addition to identifying weaknesses, suggest concrete and actionable improvements 
(e.g., clarifying assumptions, refining the method, or strengthening the evaluation plan) 
that would improve the idea's {quality_indicator} within the scope defined by the proposal.

**Proposed Research Idea:**  
{research_idea}
"""
    return prompt


def review_prompt_q():
    """
    Prompt template for generating constructive criticism of a research idea
    grounded in both the motivating questions and the proposed idea.
    """
    prompt = """You are a reviewer evaluating a research proposal.

Your task is to provide constructive criticism on the research idea's {quality_indicator}, 
helping the proposer improve it as much as possible.

Base your critique on both the motivating questions and the proposed research idea. 
Use the questions to understand the intended problem setting, underlying assumptions, 
and evaluation context implied by the proposal.

Assess whether the proposed idea is coherent and well-developed under this setting, 
and identify where gaps, ambiguities, or unrealistic assumptions limit the 
{quality_indicator}. Explain why these issues matter in the context implied by the questions.

In addition to identifying weaknesses, suggest concrete and actionable improvements 
(e.g., clarifying assumptions, refining the method, or strengthening the evaluation plan) 
that would make the idea better aligned with the problem setting suggested by the questions 
and improve its {quality_indicator}.

**Motivating Questions:**  
{question}

**Proposed Research Idea:**  
{research_idea}
"""
    return prompt




def get_idea_generation_prompt_template_generate():
    """
    Prompt template for generating a research idea using only paper abstracts.
    """
    prompt = """You are a top-tier researcher proposing research ideas. Your role is to create a research idea and refine the idea if you receive feedback. 
        A reviewer will review your research idea based on its {quality_indicator} and give you feedback. You should try your best to improve the idea based on the reviewer's feedback and your expertise, 
        especially paying attention to the idea's {quality_indicator}.

        Here is the previous idea:
        {previous_idea}
        Here is the reviewer’s feedback:
        {reviewer_feedback}

        Based on the reviewer’s feedback regarding the previous research idea’s {quality_indicator}, generate a revised and improved research idea using the following format:

        Title: [A brief, focused title]
        Problem: [The core issue or gap being addressed]
        Objective: [The main goal or research question]
        Hypothesis: [The hypothesis being tested or explored]
        Method: [The approach or methodology]
        Expected Impact/Findings: [The anticipated outcomes or contributions]

        Please only respond with the improved research idea in the format provided above. Do not respond with anything irrelevant."""
    return prompt

def get_eval_template():
    """
    Prompt template for evaluating a research idea using only paper abstracts.
    """
    prompt = """You are a reviewer tasked with ranking the quality of a set of research ideas based on their {ranking_criteria}.
        The idea with the highest {ranking_criteria} should be ranked first.

        The set of research ideas are: {research_ideas}.

        Instructions:
        - Include ALL research ideas in the ranking. Do NOT omit any.
        - Use the exact idea labels as provided (e.g., a, b, c, d, ...).
        - Provide a ranking list from best to worst based on {ranking_criteria}.
        - Include a brief rationale for each idea in the Detailed Reviews section.

        Required output format:
        Overall Ranking (from best to worst): [<labels in order>]

        Detailed Reviews:
        <label>. (brief rationale)
        <label>. (brief rationale)
        ...

        Now, please rank the following research ideas according to these instructions:
        """
    return prompt


def get_winrate_eval_template_withTie():
    prompt = """You are a reviewer tasked with comparing two sets of research ideas and determining which set is generally better.
    Your goal is to evaluate the overall quality and effectiveness of each set based on {ranking_criteria}.

    The two sets of research ideas are:
    Set A: {research_ideas_a}
    Set B: {research_ideas_b}

    Instructions:
    - Consider all ideas in both sets before making a judgment.
    - Choose the set that is better based solely on their content. Do NOT assume Set A is better because it is listed first.
    - If both sets are of similar quality and neither is better, you MUST output 'TIE'.
    - Include a brief rationale explaining why the chosen set is better, or why they are tied.
    - Optionally, you may comment on individual ideas if relevant, but focus on the set-level comparison.

    Required output format:
    Overall Winner: <Set A / Set B / TIE>

    Rationale:
    <brief explanation of the comparison>

    Now, please compare the following two sets of research ideas according to these instructions:
    """
    return prompt

def get_winrate_eval_template():
    """
    Neutral prompt template for comparing two sets of research ideas and evaluating their overall winrate.
    One set must be selected as the winner; do not favor either set due to order.
    """
    prompt = """You are a reviewer tasked with comparing two sets of research ideas and determining which set is generally better.
        Your goal is to evaluate the overall quality and effectiveness of each set based on {ranking_criteria}.

        The two sets of research ideas are:
        Set A: {research_ideas_a}
        Set B: {research_ideas_b}

        Instructions:
        - Consider all ideas in both sets before making a judgment.
        - Choose the set that is better based solely on their content. Do NOT assume Set A is better because it is listed first.
        - One set must be selected; do NOT output 'Tie'.
        - Include a brief rationale explaining why the chosen set is better.
        - Optionally, you may comment on individual ideas if relevant, but focus on the set-level comparison.

        Required output format:
        Overall Winner: <Set A / Set B>

        Rationale:
        <brief explanation of the comparison>
        
        Now, please compare the following two sets of research ideas according to these instructions:
        """
    return prompt

def get_winrate_eval_template_feasibility():
    prompt = """You are a reviewer tasked with comparing two sets of research ideas.

Your goal is to decide which set is MORE FEASIBLE as a research direction, meaning more realistic and executable overall.

The two sets of research ideas are:
Set A: {research_ideas_a}
Set B: {research_ideas_b}

Instructions:
- Consider all ideas in both sets before making a judgment.
- Judge based solely on the content. Do NOT assume Set A is better because it is listed first.
- You MUST choose either Set A or Set B as the overall winner. Do NOT output a tie.
- Do NOT reward verbosity, buzzwords, or overly ambitious promises.

When judging feasibility, prioritize the following signals:
1) Implementability: Does each idea propose a method that could realistically be implemented?
2) Explicit assumptions: Are key requirements clear or reasonably implied (data, tools, compute, supervision)?
3) Evaluation plan: Is there a plausible way to validate or test the idea (metrics, baselines, experiments)?
4) Critical blockers: Are there missing steps or unrealistic dependencies that would likely prevent execution?

Form an overall judgment by considering both:
- Typical feasibility: which set has higher average feasibility across ideas?
- Reliability: which set has fewer ideas with serious feasibility blockers?

Required output format:
Overall Winner: <Set A / Set B>

Rationale:
<brief explanation focusing on feasibility differences>

Now, please compare the two sets according to these instructions:
"""
    return prompt

def get_winrate_eval_template_novelty():
    prompt = """You are a reviewer tasked with comparing two sets of research ideas.

Your goal is to decide which set is MORE NOVEL overall, meaning more original and non-obvious as a research direction.

The two sets of research ideas are:
Set A: {research_ideas_a}
Set B: {research_ideas_b}

Instructions:
- Consider all ideas in both sets before making a judgment.
- Judge based solely on the content. Do NOT assume Set A is better because it is listed first.
- You MUST choose either Set A or Set B as the overall winner. Do NOT output a tie.
- Do NOT reward verbosity, buzzwords, or superficial claims of novelty.

When judging novelty, prioritize the following signals:
1) Non-obviousness: Do the ideas go beyond standard combinations or incremental extensions?
2) Distinct contribution: Is the core contribution clearly differentiated from common baselines?
3) Fresh perspective: Do the ideas introduce a new framing, mechanism, or hypothesis?
4) Clear gap: Is there a well-articulated gap or tension that motivates the new idea?

Form an overall judgment by considering:
- Typical novelty across ideas
- Whether one set avoids being mostly generic or incremental

Required output format:
Overall Winner: <Set A / Set B>

Rationale:
<brief explanation focusing on novelty differences>

Now, please compare the two sets according to these instructions:
"""
    return prompt


def get_context_rag_withq():
    prompt = """You are an expert AI research assistant. Your primary task is to retrieve and synthesize information.  
    To complete this task, you must actively perform **web searches** to collect the latest advancements, related concepts, and potential challenges.

    **Provided Research Papers:**  
    {papers_info}

    **Questions:**  
    {question}

    **Your Task:**  
    1. **Search the web** for relevant, up-to-date information that complements (but does not duplicate) the content of the provided research papers.  
    2. Focus your summary on insights, methods, breakthroughs, or challenges that are **absent from or not emphasized in the provided papers**, but are useful in addressing the given questions.  
    3. Generate a concise yet comprehensive context highlighting:  
    - Key technologies and frameworks  
    - Recent breakthroughs and real-world applications  
    - Open problems and common obstacles  

    This synthesized context will later be used to brainstorm a new idea, so prioritize information that adds **novel, non-overlapping, and helpful perspectives** beyond what is already contained in the research papers."""
    return prompt

def get_context_rag_withoutq():
    prompt = """You are an expert AI research assistant. Your primary task is to retrieve and synthesize information.  
    To complete this task, you must actively perform **web searches** to collect the latest advancements, related concepts, and potential challenges.

    **Provided Research Papers:**  
    {papers_info}

    **Your Task:**  
    1. **Search the web** for relevant, up-to-date information that complements (but does not duplicate) the content of the provided research papers.  
    2. Focus your summary on insights, methods, breakthroughs, or challenges that are **absent from or not emphasized in the provided papers**.  
    3. Generate a concise yet comprehensive context highlighting:  
    - Key technologies and frameworks  
    - Recent breakthroughs and real-world applications  
    - Open problems and common obstacles  

    Important: Do **not** include any URLs, links, or references to specific websites in your output.  
    This synthesized context will later be used to brainstorm a new idea, so prioritize information that adds **novel, non-overlapping, and helpful perspectives** beyond what is already contained in the research papers."""
    return prompt

def get_idea_generation_prompt_template_refine_questions():
    """
    Prompt template for refining a research idea based on multiple guiding questions and reviewer feedback.
    """
    prompt = """You are a top-tier researcher proposing research ideas. Your role is to create a research idea and refine the idea if you receive feedback. 
    A reviewer will review your research idea based on its {quality_indicator} and give you feedback. You should try your best to improve the idea based on the reviewer's feedback, your expertise, 
    and the guiding questions, especially paying attention to the idea's {quality_indicator}.

    Guiding questions:
    {questions}

    Here is the previous idea:
    {previous_idea}
    Here is the reviewer’s feedback:
    {reviewer_feedback}

    Based on the reviewer’s feedback and the guiding questions regarding the previous research idea’s {quality_indicator}, generate a revised and improved research idea using the following format:

    Title: [A brief, focused title]
    Problem: [The core issue or gap being addressed]
    Objective: [The main goal or research question]
    Hypothesis: [The hypothesis being tested or explored]
    Method: [The approach or methodology]
    Expected Impact/Findings: [The anticipated outcomes or contributions]

    Please only respond with the improved research idea in the format provided above. Do not respond with anything irrelevant."""
    return prompt