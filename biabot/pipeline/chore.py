# biabot/pipline/chore.py

import random
import re
from typing import List, Tuple, Optional

from ..constants import DOC_CHORE, DOC_ANALYSIS, DOC_NER
from .base import BasePipeline, RetryException


def chore_pipeline(self: BasePipeline, query_text: str, save: bool = True, **kwargs):
    first_step = {
        "step": 1,
        "prompt": f"""Step 1: Analyze the task: "{query_text}". Determine a course of action with 3-5 specific steps to complete this task. List these steps clearly. Begin with "Course of Action:", End with "Total Steps: [n]\n\n""",
        "max_tokens": 1024,
        'use_guidance': True,
        "document_type": DOC_CHORE,
        "top_n": 5,
        "apply_head": True,
    }

    task_descriptions = [
        "Provide a detailed description of what you're doing and any results or observations.",
        "Perform any necessary actions or tasks to complete the chore.",
        "Execute the required steps, focusing on efficiency and thoroughness.",
        "Implement the planned actions, documenting any unexpected challenges.",
        "Carry out the designated tasks, paying attention to detail and quality.",
        "Conduct the necessary operations, ensuring all objectives are met.",
        "Fulfill the outlined responsibilities, adapting to any situational changes.",
        "Complete the assigned duties, optimizing for time and resource management.",
        "Undertake the specified actions, maintaining a systematic approach.",
        "Accomplish the given tasks, demonstrating problem-solving skills when needed.",
        "Perform the required functions, prioritizing safety and effectiveness.",
        "Execute the planned activities, showcasing adaptability and initiative.",
        "Implement the necessary procedures, focusing on achieving the desired outcome.",
        "Carry out the designated operations, ensuring compliance with any relevant guidelines.",
        "Fulfill the outlined tasks, demonstrating creativity in overcoming obstacles."
    ]

    ordinal = [
        'first',
        'second',
        'third',
        'fourth',
        'fifth',
        'sixth',
        'seventh',
        'eighth',
        'ninth',
        'tenth',
        'eleventh',
        'twelfth',
        'thirteenth',
        'fourteenth',
        'fifteenth',
    ]

    intermediate_template = {
        "prompt": f"""Step %d: Execute the %s step of your course of action. %s Begin with "Executing Step %d:"\n\n""",
        "max_tokens": 512,
        "document_type": DOC_CHORE,
        "top_n": 3,
        'use_guidance': True,
        "apply_head": True,
    }

    last_step = {
        "prompt": f"""Step %d: Reflect on the entire process of completing "{query_text}". Evaluate your performance, note any challenges or insights, and suggest improvements for future tasks. Begin with "Dear Journal:"\n\n""",
        "max_tokens": 512,
        "document_type": DOC_CHORE,
        "top_n": 1,
        "retry": True,
        "apply_head": True,
    }

    self.config.system_message = self.persona.system_prompt(mood=self.config.persona_mood, location=self.config.persona_location)

    thoughts = [
        f"Task: {query_text}",
    ]
    self.prompt_prefix = f"""{self.persona.name}, this is your conscious mind. Your thoughts have brought up new memories:\n\n"""
    for thought in thoughts:
        self.prompt_prefix += f"""- {thought}\n"""

    self.filter_text = f"document_type != '{DOC_CHORE}'"

    results = self.cvm.query(
        [query_text],
        filter_doc_ids=self.seen_doc_ids,
        top_n=5,
        document_type=DOC_ANALYSIS,
    )

    step = 1
    self.accumulate(step, queries=results)

    branch = 0
    total_steps = 0

    while True:
        try:
            # Initialize the turn config
            if step == 1:
                turn_config = {**first_step}
                turn_config['step'] = step
            elif step == total_steps + 2:
                turn_config = {**last_step}
                turn_config['step'] = step
            elif step < total_steps + 2:
                turn_config = {**intermediate_template}
                step_ordinal = ordinal[step - 2]
                step_task = random.choice(task_descriptions)
                turn_config['step'] = step
                turn_config["prompt"] = turn_config["prompt"] % (
                    step,
                    step_ordinal,
                    step_task,
                    step,
                )
            else:
                break
            turn_config['branch'] = branch

            # Execute the turn and get the response
            response = self.execute_turn(**turn_config)

            # We need to get the total steps
            if total_steps == 0:
                extracted = self.patterns.extract_total_steps(response)
                total_steps = extracted if extracted is not None else 0
            
            if total_steps == 0:
                original_prompt = turn_config["prompt"]
                turn_config["prompt"] = "Please output the total steps. Begin with 'Total Steps:'\n\n"
                new_response = self.execute_turn(**turn_config)
                extracted = self.patterns.extract_total_steps(new_response)
                total_steps = extracted if extracted is not None else 0
                turn_config["prompt"] = original_prompt
                response = f"{response}\n\n{new_response}"
                    
            if total_steps == 0:
                raise ValueError("Could not determine total steps.")
                
            # Accept and continue
            self.accept_response(response=response, **turn_config)

            step += 1
        except RetryException:
            continue
        except StopIteration:
            break

    return
