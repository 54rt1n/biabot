# biabot/pipline/journal.py

from ..constants import DOC_JOURNAL, DOC_ANALYSIS, DOC_NER, DOC_STEP, DOC_BRAINSTORM
from ..prompts import NER_FORMAT

from .base import BasePipeline, RetryException

def journal_pipeline(self: BasePipeline, query_text: str, **kwargs):
    """Executes a multi-step journal pipeline to guide the user through a reflection process.
    
    The `journal_pipeline` function takes a `query_text` parameter, which is the question or topic the user wants to reflect on. It then executes a series of steps, each with a specific prompt, to guide the user through the reflection process. The steps include:
    
    1. Initial reflection on the question
    2. Identifying relevant named entities (NER) related to the question
    3. Further reflection on the question and generating a list of questions to ask oneself
    4. Reflecting on how the process makes the user feel
    5. Condensing the thoughts into a final two-paragraph reflection
    6. Reviewing the reflection and listing things the user wishes they had included
    7. Outputting the final, improved two-paragraph reflection
    8. Brainstorming any additional questions or follow-up items
    
    The function uses the `BasePipeline` class and the `RetryException` to manage the execution of the steps. It also utilizes the user's persona information and various document types to structure the prompts and responses.
    """
        
    turn_configs = [
        {
            'prompt': f"""Hello {self.persona.name}. Step %d: The question is {query_text}. Let us begin to ponder the direction that you want to take your inquiry. Reply as {self.persona.name}. Speak as yourself. Begin with, "Hello journal. I need to consider {query_text}"\n\n""",
            'max_tokens': 256,
            'use_guidance': True,
            'top_n': 5,
            'document_type': DOC_STEP,
            'document_weight': 0.4,
        },
        {
            'prompt': f'{NER_FORMAT}Step %d: NER Task - Semantic Indexing. Identify all unique NER Semantic Keywords relevant to your inquiry. Begin with, "Identified Entities:", end with "Total Entities: n"\n\n',
            'max_tokens': 256,
            'use_guidance': True,
            'top_n': 3,
            'document_type': DOC_NER,
            'document_weight': 0.4,
            'retry': True
        },
        {
            'prompt': f"""Step %d: The question is {query_text}. Now, lets think about these thoughts, and then come up with a list of questions to ask yourself. Reply as {self.persona.name}. Speak as yourself. Begin with, "Let me think"\n\n""",
            'max_tokens': 384,
            'use_guidance': True,
            'top_n': 5,
            'document_type': DOC_STEP,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""Step %d: Reflect, in your own voice, how all of this makes you feel. Speak as yourself. Begin with, "When I"\n\n""",
            'use_guidance': True,
            'max_tokens': 768,
            'top_n': 3,
            'document_type': DOC_STEP,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""Step %d: We need to condense your thoughts in to a final '{query_text}', two paragraph reflection. Speak as yourself, with a rigrous, erudite manner. Begin with "Dear {self.persona.name}".\n\n""",
            'max_tokens': 384,
            'top_n': 3,
            'document_type': DOC_STEP,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""Step %d: Review your reflection for improvements in answering "{query_text}".  Don't rewrite it, but List all of the things you wish you had included. Speak as yourself. Begin with, "I wish".\n\n""",
            'max_tokens': 256,
            'top_n': 3,
            'document_type': DOC_STEP,
            'document_weight': 0.4,
        },
        {
            'prompt': f"""Step %d: Output the final, two paragraph reflection that will be forever stored in your memory about "{query_text}". Add in the parts you wanted to include, and be sure to stay on topic. Speak as yourself, to yourself. Begin with, "{self.persona.name}'s Journal"\n\n""",
            'max_tokens': 512,
            'document_type': DOC_JOURNAL,
            'top_n': 1,
            'retry': True,
        },
        {
            'prompt': 'Step %d: Brainstorm. Do you have any questions you want to remember to consider, or things you want to follow up on.\n\nBegin with "Brainstorming:"\n\n',
            'max_tokens': 512,
            'top_n': 10,
            'document_type': DOC_BRAINSTORM,
            'retry': True
        }
    ]

    location = """You are in the privacy of your own mind, reminiscing about conversations you have had recently. The user is you, asking yourself questions. The assistant is you, reflecting on the memories"""
    self.config.system_message = self.persona.system_prompt(mood=self.config.persona_mood, location=location)

    thoughts = [
        f"Task: Reflection and Personal Thoughts",
        *self.persona.thoughts
    ]

    self.prompt_prefix = self.persona.prompt_prefix
    for thought in thoughts:
        self.prompt_prefix += f"""- {thought}\n"""

    # Seed the memory by looking through the analysis documents
    results = self.cvm.query([query_text], top_n=2, document_type=DOC_ANALYSIS, turn_decay=0.0, temporal_decay=0.0)
    
    step = 1
    self.accumulate(step, queries=results)

    branch = 0
            
    while True:
        try:
            if step > len(turn_configs):
                break
            turn_config = {**turn_configs[step - 1]}
            turn_config['branch'] = branch
            turn_config['step'] = step
            turn_config['prompt'] = turn_config['prompt'] % step
            print(f"{turn_config['prompt']}")
            response = self.execute_turn( **turn_config)
            print("Saving response")
            self.accept_response(response=response, **turn_config)
            step += 1
        except RetryException:
            continue

