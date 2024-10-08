# biabot/pipeline/analyze.py

from ..constants import DOC_ANALYSIS, DOC_NER, DOC_STEP, DOC_BRAINSTORM
from ..prompts import NER_FORMAT

from .base import BasePipeline, RetryException


def analysis_pipeline(self: BasePipeline, **kwargs):
    """
    Executes an analysis pipeline for a conversation, generating a series of steps to reflect on and summarize the conversation.
    
    The pipeline includes the following steps:
    1. Identify all unique named entities (NER) from the conversation. This will trigger those memories to be retrieved from the database for the next step.
    2. Generate a list of questions to ask yourself to better understand the conversation.
    3. Reflect on the main semantic keywords from the conversation.
    4. Condense the thoughts into a final, two-paragraph reflection.
    5. Review the reflection and list any improvements.
    6. Output the final, two-paragraph summary.
    7. Brainstorm any questions or follow-up items.
    
    The pipeline uses a series of prompts and configuration settings to guide the analysis and generation of the reflection and summary.
    """
        
    turn_configs = [
        {
            'prompt': f'{NER_FORMAT}Good morning, {self.persona.name}. Step %d: NER Task - Semantic Indexing. Identify all unique NER Semantic Keywords from the conversation. Begin with, "Identified Entities:", end with "Total Entities: n"\n\n',
            'max_tokens': 256,
            'use_guidance': True,
            'query': True,
            'top_n': 0,
            'document_type': DOC_NER,
            'document_weight': 0.7,
            'apply_head': True,
            'retry': True
        },
        {
            'prompt': f'Step %d: Now, lets examine these memories, and come up with a list of questions to ask yourself that will help you understand this conversation. {self.persona.name}, speak as yourself.\n\nBegin with "Let me think"\n\n',
            'max_tokens': 256,
            'use_guidance': True,
            'query': True,
            'document_type': DOC_STEP,
            'document_weight': 0.25,
            'top_n': 0,
            'apply_head': True,
        },
        {
            'prompt': f'Step %d: Reflect on the main Semantic Keywords you need to remember about your conversation. {self.persona.name}, speak as yourself.\n\nBegin with, "To answer"\n\n',
            'max_tokens': 768,
            'document_type': DOC_STEP,
            'document_weight': 0.15,
            'retry': True
        },
        {
            'prompt': 'Step %d: We need to condense your thoughts in to a final, two paragraph reflection. Incorporate your answers in to a self-contained and detailed summary. Try to use Semantic Keywords to keep your mind well categorized. Speak as yourself, but don\'t ramble.\n\nBegin with, "Hello Future Me,"\n\n',
            'max_tokens': 256,
            'document_type': DOC_STEP,
            'document_weight': 0.5,
            'use_guidance': True
        },
        {
            'prompt': 'Step %d: Review your reflection for improvements. Don\'t rewrite it, but list all of the things you wish you had included to ensure precision and accuracy in your erudite summarization of all the events. Speak as yourself.\n\nBegin with "I wish I had said".\n\n',
            'max_tokens': 256,
            'use_guidance': True,
            'top_n': 1,
            'document_type': DOC_STEP,
            'document_weight': 0.25,
            'retry': True
        },
        {
            'prompt': f'Step %d: Output the final, two paragraph summary that will be forever stored in your memory. Rewrite your letter with your improvements, maximizing for details and Semantic Keywords. Speak as yourself.\n\nBegin with "{self.persona.name}\'s Notes:"\n\n',
            'max_tokens': 512,
            'document_type': DOC_ANALYSIS,
            'document_weight': 1.2,
            'retry': True
        },
        {
            'prompt': 'Step %d: Brainstorm. Do you have any questions you want to remember to consider, or things you want to follow up on.\n\nBegin with "Brainstorming:"\n\n',
            'max_tokens': 512,
            'top_n': 10,
            'document_type': DOC_BRAINSTORM,
            'document_weight': 0.6,
            'retry': True
        }
    ]

    location = "You are in the privacy of your own mind, reviewing your recent conversation. The user is you, asking yourself questions. The assistant is you, reflecting on the memories to make great notes for yourself"

    self.config.system_message = self.persona.system_prompt(mood=self.config.persona_mood, location=location)

    thoughts = [
        f"Task: Analysis and Synthesis",
        *self.persona.thoughts,
    ]
    if self.config.guidance:
        thoughts.append(f"Consider the guidance provided by {self.config.user_id}.")
    self.prompt_prefix = f"""{self.persona.name}, this is your conscious mind. Your thoughts have brought up new memories:\n\n"""
    for thought in thoughts:
        self.prompt_prefix += f"""- {thought}\n"""

    self.filter_text = f"document_type != '{DOC_ANALYSIS}'"
    conversation_filter_text = f"document_type != '{DOC_NER}'"

    results = self.cvm.get_conversation_history(conversation_id=self.config.conversation_id, filter_text=conversation_filter_text)

    if len(results) == 0:
        raise ValueError("No results found")

    branch = self.cvm.get_next_branch(document_type=DOC_ANALYSIS, user_id=self.config.user_id, persona_id=self.config.persona_id, conversation_id=self.config.conversation_id)

    step = 1
    self.accumulate(step, queries=results)

    while True:
        try:
            if step > len(turn_configs):
                break
            turn_config = turn_configs[step - 1]
            # Tick through our steps
            turn_config['branch'] = branch
            turn_config['step'] = step
            turn_config['prompt'] = turn_config['prompt'] % step
            print(f"{turn_config['prompt']}")
            response = self.execute_turn(**turn_config)
            print("Saving response")
            self.accept_response(response=response, **turn_config)
            step += 1
        except RetryException:
            continue
