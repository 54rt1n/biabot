# biabot/pipline/chore.py

import random
from typing import Optional

from ..constants import DOC_NER, DOC_REPORT, DOC_ANALYSIS, DOC_STEP
from ..prompts import NER_FORMAT
from .base import BasePipeline, RetryException

def build_document(self: BasePipeline, filename: str) -> str:
    """
    Builds a formatted document containing the contents of the specified file.
    
    Args:
        self (BasePipeline): The pipeline instance.
        filename (str): The name of the file to read.
    
    Returns:
        str: The formatted document contents.
    """
        
    print(f"Current Document: {filename}")
    document_contents = self.library.read_document(filename)
    content = []
    content.append(f"==={self.config.user_id} is revealing a Document to you===")
    content.append(f"Document Name: [{filename}]")
    content.append(f"Document Length: {len(document_contents)}")
    content.append("===Begin Document===")
    content.append(document_contents)
    content.append("===End of Document===")
    #print(f"Document Contents: {content}")

    return '\n'.join(content)

def report_pipeline(self: BasePipeline, query_text: str, document_name: Optional[str] = None, **kwargs):
    """
    Generates a report pipeline for a given query text and optional document.
    
    The report pipeline performs the following steps:
    1. Retrieves the contents of the specified document, if provided.
    2. Generates a prompt for an NER (Named Entity Recognition) task to identify relevant semantic keywords.
    3. Generates a prompt for an analysis task to determine a course of action with 3-5 steps.
    4. Generates a series of intermediate steps to execute the planned course of action.
    5. Generates a draft report for the given query text.
    6. Generates a review of the draft report to identify areas for improvement.
    7. Generates the final report for the given query text.
    
    The report pipeline utilizes the BasePipeline class and various configuration parameters to control the behavior of the tasks.
    
    Args:
        self (BasePipeline): The pipeline instance.
        query_text (str): The text of the query to generate the report for.
        document_name (Optional[str]): The name of the document to include in the report, if any.
    
    Returns:
        None
    """
        
    addin = ""
    if document_name is not None:
        document_contents = build_document(self, document_name)
        addin = f' and pay close attention to the provided Document. Begin with, "Identified Entities [{document_name}]:'
    else:
        document_contents = ""
        addin = '. Begin with, "Identified Entities:'
    ner_task = {
            'prompt': f'{NER_FORMAT}Step %d: NER Task - Semantic Indexing. Identify all unique NER Semantic Keywords relevant to your inquiry, drawing from your memories{addin}, end with "Total Entities: n"\n\n',
            'max_tokens': 256,
            'use_guidance': True,
            'document_type': DOC_NER,
            'document_weight': 0.4,
            'retry': True
        }
    analysis_task = {
        "prompt": f"""Step %d: Analyze the task: "{query_text}". Determine a course of action with 3-5 steps to draw and summarize the relevant information from your Active Memory. List these steps clearly. Begin with "Synthesis Plan:", End with "Total Steps: [n]\n\n""",
        "max_tokens": 1024,
        "document_type": DOC_STEP,
        "top_n": 3,
        'use_guidance': True,
        "weight": 0.3,
        "temporal_decay": 0.5,
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
        'use_guidance': True,
        "document_type": DOC_STEP,
        "top_n": 2,
        "weight": 0.3,
        "apply_head": False,
    }

    draft_report = {
        "prompt": f"""Step %d: It is time to generate your draft report about "{query_text}". Be thourough but concise. Begin with "Draft Report:"\n\n""",
        "max_tokens": 1024,
        "document_type": DOC_STEP,
        "top_n": 1,
        "weight": 0.7,
        "retry": True,
        "apply_head": False,
    }

    report_review = {
        "prompt": f"""Step %d: Review your draft report and highlight any areas that need improvement. Begin with "I wish I had"\n\n""",
        "max_tokens": 384,
        "document_type": DOC_STEP,
        "top_n": 1,
        "weight": 0.3,
        "retry": True,
        "apply_head": False,
    }

    final_report = {
        "prompt": f"""Step %d: {self.persona.name}, it is time to generate your final report about "{query_text}". Be thourough but concise. Begin with "Dear Sir:"\n\n""",
        "max_tokens": 1024,
        "document_type": DOC_REPORT,
        "top_n": 1,
        "weight": 1.3,
        "retry": True,
        "apply_head": False,
    }

    self.config.system_message = self.persona.system_prompt(mood=self.config.persona_mood, location=self.config.persona_location)

    thoughts = [
        f"Task: Report on {query_text}",
        *self.persona.thoughts
    ]
    self.prompt_prefix = self.persona.prompt_prefix
    for thought in thoughts:
        self.prompt_prefix += f"""- {thought}\n"""

    self.prompt_prefix += document_contents

    # Chunk our document in to 128 word chunks, which should be about 400 tokens; and we will use those for our search
    chunk_size = 128
    doc = document_contents.split(' ')
    chunks = [
        ' '.join(doc[i:i + chunk_size])
        for i in range(0, len(doc), chunk_size)
    ]

    # TODO this should filter for relevancy instead
    top_n = 3 if len(chunks) < 2 else 2
    results = self.cvm.query(
        [query_text, *chunks],
        top_n=top_n,
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
                turn_config = {**ner_task}
                turn_config['step'] = step
                turn_config["prompt"] = turn_config["prompt"] % (step)
            elif step == 2:
                turn_config = {**analysis_task}
                turn_config['step'] = step
                turn_config["prompt"] = turn_config["prompt"] % (step)
            elif step < total_steps + 3:
                turn_config = {**intermediate_template}
                step_ordinal = ordinal[step - 3]
                step_task = random.choice(task_descriptions)
                turn_config['step'] = step
                turn_config["prompt"] = turn_config["prompt"] % (
                    step,
                    step_ordinal,
                    step_task,
                    step,
                )
            elif step == total_steps + 3:
                turn_config = {**draft_report}
                turn_config['step'] = step
                turn_config["prompt"] = turn_config["prompt"] % (step)
            elif step == total_steps + 4:
                turn_config = {**report_review}
                turn_config['step'] = step
                turn_config["prompt"] = turn_config["prompt"] % (step)
            elif step == total_steps + 5:
                turn_config = {**final_report}
                turn_config['step'] = step
                turn_config["prompt"] = turn_config["prompt"] % (step)
            else:
                break
            turn_config['branch'] = branch

            # Execute the turn and get the response
            response = self.execute_turn(**turn_config)

            if step > 1:
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
                    # Guess
                    total_steps = 4
                    #raise ValueError("Could not determine total steps.")
                    
            # Accept and continue
            self.accept_response(response=response, **turn_config)

            step += 1
        except RetryException:
            continue
        except StopIteration:
            break

    return
