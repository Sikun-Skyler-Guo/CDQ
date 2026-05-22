# openai_client.py
# Encapsulates all interactions with the OpenAI API and DeepInfra API

import os
import warnings
from openai import OpenAI
from config import OPENAI_API_KEY, DEEPINFRA_API_KEY
import re

# Suppress the thread._ident warning from OpenAI SDK
# This is a known warning in multi-threaded environments that doesn't affect functionality
warnings.filterwarnings("ignore", message=".*thread._ident.*")

def extract_ranking(text):
    if text is None or (isinstance(text, float) and str(text).lower() == 'nan'):
        return None
    match = re.search(r"\[(.*?)\]", str(text))
    if match:
        return match.group(1).strip()
    return None

class OpenAIClient:
    def __init__(self, model="gpt-4o", api_provider="openai"):
        """
        Initialize the API client (OpenAI or DeepInfra).
        - Check if the API key exists.
        - Create an OpenAI-compatible client instance.
        
        Args:
            model: Model name to use (default: "gpt-4o", can be "gpt-4o-mini" or other models)
            api_provider: API provider to use - "openai" or "deepinfra" (default: "openai")
        """
        self.api_provider = api_provider.lower()
        self.model = model  # Store model name
        
        if self.api_provider == "openai":
            if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
                raise ValueError("OpenAI API key not found. Please set it in config.py")
            # Initialize OpenAI client with timeout settings to avoid thread issues
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
                timeout=300.0,
                max_retries=2
            )
        elif self.api_provider == "deepinfra":
            if not DEEPINFRA_API_KEY or DEEPINFRA_API_KEY == "YOUR_DEEPINFRA_API_KEY":
                raise ValueError("DeepInfra API key not found. Please set it in config.py")
            # Initialize DeepInfra client (OpenAI-compatible API)
            # DeepInfra uses OpenAI-compatible API with a different base_url
            # Note: The thread._ident warning may appear but doesn't affect functionality
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.client = OpenAI(
                    api_key=DEEPINFRA_API_KEY,
                    base_url="https://api.deepinfra.com/v1/openai",
                    timeout=300.0,
                    max_retries=2
                )
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}. Use 'openai' or 'deepinfra'")
    
    def _format_papers_info(self, paper):
        """
        Format a list of papers into a single string.
        """
        info_str = ""
        info_str += "References:\n"
        for j, ref_title in enumerate(paper["references"], 1):
            info_str += f"  {j}. {ref_title}\n"
        return info_str
    
    def _format_papers_info_baseline(self, paper):
        """
        Format a list of papers into a single string.
        """
        info_str = ""
        info_str += "References:\n"
        for j, ref_title in enumerate(paper["references_abs"], 1):
            info_str += f"  abstract: {paper['references_abs'][j-1]}\n"
        return info_str
    
    def _format_questions(self, paper):
        """
        Format a list of papers into a single string.
        """
        info_str = ""
        info_str += "Questions:\n"
        for j, question in enumerate(paper["questions"], 1):
            info_str += f"  {j}. : {question}\n"
        return info_str
    
    def _format_research_ideas(self, paper):
        info_str = ""
        info_str += "Research Ideas:\n"
        info_str += f" Research idea A: {paper['idea_a']}\n"
        info_str += f" Research idea B: {paper['idea_b']}\n"
        info_str += f" Research idea C: {paper['idea_c']}\n"
        info_str += f" Research idea D: {paper['idea_d']}\n"
        return info_str
    
    def _format_research_ideas_winrate(self, paper, prefix=""):
        info_str = "Research Ideas:\n"
        for i, key in enumerate(["idea_b", "idea_c", "idea_d"], start=1):
            full_key = f"{prefix}{key}"
            info_str += f" {i}. {paper.get(full_key, '')}\n"
        return info_str
    
    def _format_questions_info(self, paper):
        """
        Format a list of papers into a single string.
        """
        info_str = ""
        for j, ref_title in enumerate(paper["questions"], 1):
            info_str += f"  {j}. {ref_title}\n"
        return info_str
    
    def generate_context_with_rag_without_question(self, papers, rag_prompt_template):
        papers_info = self._format_papers_info(papers)
        
        # Fill in the prompt template
        final_prompt = rag_prompt_template.format(papers_info=papers_info)

        # # save money, no using api-keys right now.
        # return final_prompt
        if self.api_provider == "openai":
            return self._generate_context_with_rag_without_question_openai(final_prompt)
        elif self.api_provider == "deepinfra":
            return self._generate_context_with_rag_without_question_deepinfra(final_prompt)
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
    
    def _generate_context_with_rag_without_question_openai(self, final_prompt):
        """Generate context using OpenAI API with web search."""
        try:
            print(f"Calling OpenAI API to generate context (x1) using {self.model}...")
            response = self.client.responses.create(
                model=self.model,  # Use the specified model
                tools= [{ "type": "web_search" }],
                input=[
                    {"role": "system", "content": "You are a helpful research assistant with access to web knowledge."},
                    {"role": "user", "content": final_prompt}
                ]
            )
            context = response.output_text
            return context.strip()
        except Exception as e:
            print(f"Error while calling OpenAI API: {e}")
            return None
    
    def _generate_context_with_rag_without_question_deepinfra(self, final_prompt):
        """Generate context using DeepInfra API (no web search support, use regular chat completion)."""
        try:
            print(f"Calling DeepInfra API to generate context (x1) using {self.model}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant with access to web knowledge."},
                    {"role": "user", "content": final_prompt}
                ]
            )
            context = response.choices[0].message.content
            return context.strip()
        except Exception as e:
            print(f"Error while calling DeepInfra API: {e}")
            return None


    def generate_context_with_rag(self, papers, rag_prompt_template, withQ):
        """
        Generate context (x1) using RAG.
        This simulates the process of augmenting information via web search.
        Modern OpenAI models (e.g., gpt-4o) can leverage internal knowledge and reasoning 
        even without explicit tool calls. This prompt is designed to guide the model in that direction.

        Args:
            papers (list): List of paper information (x0).
            question (str): User question (q).
            rag_prompt_template (str): RAG prompt template (p).

        Returns:
            str: Generated context (x1).
        """
        papers_info = self._format_papers_info_baseline(papers)

        if withQ:
            questions_info = self._format_questions_info(papers)
            # Fill in the prompt template
            final_prompt = rag_prompt_template.format(papers_info=papers_info, question=questions_info)
        else:
            final_prompt = rag_prompt_template.format(papers_info=papers_info)

        # # save money, no using api-keys right now.
        # return final_prompt
        if self.api_provider == "openai":
            return self._generate_context_with_rag_openai(final_prompt)
        elif self.api_provider == "deepinfra":
            return self._generate_context_with_rag_deepinfra(final_prompt)
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
    
    def _generate_context_with_rag_openai(self, final_prompt):
        """Generate context using OpenAI API with web search."""
        try:
            print(f"Calling OpenAI API to generate context (x1) using {self.model}...")
            response = self.client.responses.create(
                model=self.model,  # Use the specified model
                tools= [{ "type": "web_search" }],
                input=[
                    {"role": "system", "content": "You are a helpful research assistant with access to web knowledge."},
                    {"role": "user", "content": final_prompt}
                ]
            )
            context = response.output_text
            return context.strip()
        except Exception as e:
            print(f"Error while calling OpenAI API: {e}")
            return None
    
    def _generate_context_with_rag_deepinfra(self, final_prompt):
        """Generate context using DeepInfra API (no web search support, use regular chat completion)."""
        try:
            print(f"Calling DeepInfra API to generate context (x1) using {self.model}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant with access to web knowledge."},
                    {"role": "user", "content": final_prompt}
                ]
            )
            context = response.choices[0].message.content
            return context.strip()
        except Exception as e:
            print(f"Error while calling DeepInfra API: {e}")
            return None

    # placeholder for idea generation module
    def generate_idea(self, papers, context, question, idea_prompt_template):
        """
        Generate the final research idea.

        Args:
            papers (list): List of paper information (x0).
            context (str): RAG-generated context (x1).
            question (str): User question (q).
            idea_prompt_template (str): Idea generation prompt template.

        Returns:
            str: The final generated idea.
        """
        papers_info = self._format_papers_info(papers)

        # Fill in the prompt template
        final_prompt = idea_prompt_template.format(
            papers_info=papers_info, 
            context=context, 
            question=question
        )
        return self.generate_idea_part(final_prompt)

    def generate_idea_withoutQ_single(self, papers, idea_prompt_template, indicator):
        papers_info = self._format_papers_info_baseline(papers)

        # Fill in the prompt template
        final_prompt = idea_prompt_template.format(
            title_abstract_pair=papers_info, 
            quality_indicator=indicator, 
        )
        return self.generate_idea_part(final_prompt)

    def generate_idea_withQ_single(self, papers, idea_prompt_template, indicator):
        papers_info = self._format_papers_info_baseline(papers)
        questions_info = self._format_questions(papers)

        # Fill in the prompt template
        final_prompt = idea_prompt_template.format(
            title_abstract_pair=papers_info, 
            quality_indicator=indicator, 
            questions=questions_info
        )
        return self.generate_idea_part(final_prompt)

    def generate_idea_rag(self, papers, context, idea_prompt_template, indicator):
        papers_info = self._format_papers_info_baseline(papers)

        # Fill in the prompt template
        final_prompt = idea_prompt_template.format(
            title_abstract_pair=papers_info, 
            rag_context=context,
            quality_indicator=indicator
        )
        return self.generate_idea_part(final_prompt)

    def generate_idea_withQ_rag(self, papers, context, idea_prompt_template, indicator):
        papers_info = self._format_papers_info_baseline(papers)
        questions_info = self._format_questions(papers)

        # Fill in the prompt template
        final_prompt = idea_prompt_template.format(
            title_abstract_pair=papers_info, 
            rag_context=context,
            quality_indicator=indicator, 
            questions=questions_info
        )
        return self.generate_idea_part(final_prompt)

    def generate_review_noq(self, idea, reviewer_prompt, indicator):
        # Fill in the prompt template
        final_prompt = reviewer_prompt.format(
            quality_indicator=indicator, 
            research_idea=idea
        )
        return self.generate_idea_part(final_prompt)
    
    def generate_review_q(self, paper, idea, reviewer_prompt, indicator):
        # Fill in the prompt template
        questions_info = self._format_questions(paper)
        final_prompt = reviewer_prompt.format(
            quality_indicator=indicator, 
            question=questions_info,
            research_idea=idea
        )
        return self.generate_idea_part(final_prompt)
    
    def generate_idea_withoutQ_generator(self, previous_idea, reviewer_feedback, generator_prompt, indicator):
        # Fill in the prompt template
        final_prompt = generator_prompt.format(
            quality_indicator=indicator, 
            previous_idea=previous_idea,
            reviewer_feedback=reviewer_feedback
        )
        return self.generate_idea_part(final_prompt)

    def generate_idea_withQ_generator(self, paper, previous_idea, reviewer_feedback, generator_prompt, indicator):
        questions_info = self._format_questions(paper)
        # Fill in the prompt template
        final_prompt = generator_prompt.format(
            quality_indicator=indicator, 
            previous_idea=previous_idea,
            reviewer_feedback=reviewer_feedback,
            questions=questions_info
        )
        return self.generate_idea_part(final_prompt)

    def generate_score(self, paper, eval_template, indicator):
        papers_info = self._format_research_ideas(paper)

        # Fill in the prompt template
        final_prompt = eval_template.format(
            research_ideas=papers_info, 
            ranking_criteria=indicator, 
        )
        for _ in range(10):
            res = self.generate_idea_part(final_prompt)
            match = re.search(r"\[(.*?)\]", str(res))
            if match:
                ranking = match.group(1).strip()
                items = [x.strip() for x in ranking.split(",")]
                if len(items) == 4: 
                    return res  
        return None
    
    def generate_winrate(self, paper, eval_template, indicator):
        papers_info_noq = self._format_research_ideas_winrate(paper, prefix="")
        papers_info_q = self._format_research_ideas_winrate(paper, prefix="q_")

        # Fill in the prompt template
        final_prompt = eval_template.format(
            research_ideas_a=papers_info_noq, 
            research_ideas_b=papers_info_q, 
            ranking_criteria=indicator, 
        )
        all_results = []
        winners_list = []  # Store winner for each iteration
        set_a_count = 0
        set_b_count = 0
        
        # Generate 5 valid results (retry if format doesn't match)
        max_attempts = 10  # Maximum attempts to get 5 valid results
        attempts = 0
        
        while len(winners_list) < 5 and attempts < max_attempts:
            attempts += 1
            res = self.generate_idea_part(final_prompt)
            if res is None:
                continue
            
            res_str = str(res).strip()
            res_lower = res_str.lower()
            
            # Check if result starts with "Overall Winner: Set A" or "Overall Winner: Set B"
            if res_lower.startswith("overall winner: set a"):
                winner = "A"
                set_a_count += 1
                all_results.append(res_str)
                winners_list.append(winner)
            elif res_lower.startswith("overall winner: set b"):
                winner = "B"
                set_b_count += 1
                all_results.append(res_str)
                winners_list.append(winner)
            # If format doesn't match, skip and retry (don't add to results)
        
        if not all_results:
            return None, None, []
        
        # Determine final winner based on majority
        if set_a_count > set_b_count:
            final_winner = "Set A"
        elif set_b_count > set_a_count:
            final_winner = "Set B"
        else:
            final_winner = "Tie"
        
        # Return: (final_result_text, final_winner, winners_list)
        return all_results[-1] if all_results else None, final_winner, winners_list
        # res = self.generate_idea_part(final_prompt)
        # return res

    def generate_idea_part(self, final_prompt):
        """
        Generate idea using the configured API provider (OpenAI or DeepInfra).
        
        Args:
            final_prompt: The prompt to send to the API
            
        Returns:
            str: Generated idea text, or None if error
        """
        if self.api_provider == "openai":
            return self._generate_idea_part_openai(final_prompt)
        elif self.api_provider == "deepinfra":
            return self._generate_idea_part_deepinfra(final_prompt)
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
    
    def _generate_idea_part_openai(self, final_prompt):
        """Generate idea using OpenAI API."""
        try:
            print(f"\nCalling OpenAI API to generate final idea using {self.model}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a creative researcher."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7  # Slightly increase creativity
            )
            idea = response.choices[0].message.content
            return idea.strip()
        except Exception as e:
            print(f"Error while calling OpenAI API: {e}")
            return None
    
    def _generate_idea_part_deepinfra(self, final_prompt):
        """Generate idea using DeepInfra API."""
        try:
            print(f"\nCalling DeepInfra API to generate final idea using {self.model}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a creative researcher."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7  # Slightly increase creativity
            )
            idea = response.choices[0].message.content
            return idea.strip()
        except Exception as e:
            print(f"Error while calling DeepInfra API: {e}")
            return None
