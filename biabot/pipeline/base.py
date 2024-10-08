# biabot/pipeline.py

from dataclasses import asdict
import os
import pandas as pd
from typing import List, Dict, Optional, Set

from ..config import ChatConfig
from ..constants import DOC_ANALYSIS, DOC_NER, LISTENER_SELF, ROLE_ASSISTANT, ROLE_USER
from ..documents import Library
from ..llm import LLMProvider
from ..models.conversation import ConversationModel
from ..patterns import Patterns
from ..persona import Persona


class RetryException(Exception):
    pass


class BasePipeline:
    """
    The `BasePipeline` class is the base class for implementing a conversational pipeline in the BIA Bot system. It provides functionality for managing the conversation history, querying relevant memories, generating responses, and accepting responses.
    
    The `BasePipeline` class has the following key responsibilities:
    - Initializing the pipeline with the necessary components (LLM provider, conversation model, persona, and configuration)
    - Generating responses by streaming turns from the language model and concatenating the results
    - Accumulating the results of memory queries into the conversation history
    - Formatting the conversation history for the current step and the complete conversation
    - Cycling the "conscious" memories, which represent the current state of the bot's internal consciousness
    - Querying the conversation model for relevant memories based on the user's prompt
    - Executing a single turn of the conversation, including querying memories, formatting the history, and generating a response
    - Accepting a generated response and appending it to the conversation history
    
    The `BasePipeline` class can be instantiated using the `from_config` class method, which constructs a new instance of the class from the provided configuration settings.
    """
        
    def __init__(self, llm: LLMProvider, cvm: ConversationModel, persona: Persona, config: ChatConfig):
        """
        The `__init__` method initializes the `BasePipeline` class with the necessary components for managing the conversation history, querying relevant memories, generating responses, and accepting responses.
        
        Args:
            llm (LLMProvider): The language model provider for generating responses.
            cvm (ConversationModel): The conversation model for querying relevant memories.
            persona (Persona): The persona of the assistant.
            config (ChatConfig): The chat configuration.
        
        The method sets up the following instance variables:
        - `llm`: The language model provider.
        - `cvm`: The conversation model.
        - `persona`: The persona of the assistant.
        - `config`: The chat configuration.
        - `patterns`: The patterns used for the pipeline.
        - `library`: The library of documents used in the pipeline.
        - `turns`: The list of turns in the conversation.
        - `history`: The history of the conversation.
        - `conscious`: The "conscious" memories, which represent the current state of the bot's internal consciousness.
        - `prompt_prefix`: The prefix for the prompt.
        - `filter_text`: An optional text filter for the conversation.
        """
                
        self.llm = llm
        self.cvm = cvm
        self.persona = persona
        self.config = config
        self.patterns = Patterns(config=config)
        self.library = Library(documents_dir=config.documents_dir)
        self.turns : List[Dict[str, str]] = []
        self.history : Dict[int, List[Dict[str, str]]] = {}
        self.conscious : List[Dict[str, str]] = []
        self.prompt_prefix = ""
        self.filter_text : Optional[str] = None

    def generate_response(self, turns: List[Dict[str, str]], config: ChatConfig) -> str:
        """
        Generates a response by streaming turns from the language model and concatenating the results.
        
        Args:
            turns (List[Dict[str, str]]): A list of turns, where each turn is a dictionary containing the speaker, role, and content.
            config (ChatConfig): The chat configuration.
        
        Returns:
            str: The generated response.
        """
                
        chunks = []
        print(f"Assistant: ", end='', flush=True)
        for t in self.llm.stream_turns(turns, config):
            if t is not None:
                print(t, end='', flush=True)
                chunks.append(t)
            else:
                print('', flush=True)
        response = ''.join(chunks)
        return response

    def accumulate(self, step: int, queries: pd.DataFrame, apply_head: bool = False, date_sort: bool = True, **kwargs):
        """
        Accumulates the results of a memory query into the history of the conversation.
        
        Args:
            step (int): The current step or turn of the conversation.
            queries (pd.DataFrame): A DataFrame containing the results of the memory query.
            apply_head (bool, optional): If True, the new entries will be added to the beginning of the history for the given step. Defaults to False.
            date_sort (bool, optional): If True, the history entries will be sorted by date in ascending order. Defaults to True.
        
        Returns:
            None
        """
                
        print(f"Results: {len(queries)}")
        for i, r in queries.iterrows():
            print(f"{i}: {r['doc_id']}/{r['conversation_id']}/{r['document_type']}: {r['date']}")
        queries = queries[['doc_id', 'document_type', 'conversation_id', 'date', 'speaker', 'role', 'content']]
        new_entries = [r.to_dict() for _, r in queries.iterrows()]
        if apply_head:
            self.history[step] = new_entries + self.history.get(step, [])
        else:
            self.history[step] = self.history.get(step, []) + new_entries

        if date_sort:
            self.history[step] = sorted(self.history[step], key=lambda x: x['date'], reverse=False)

    def format_history(self, step: int) -> str:
        """
        Formats the conversation history for the current step.
        
        Args:
            step (int): The current step or turn of the conversation.
        
        Returns:
            str: A formatted string representing the conversation history for the given step.
        """
                
        user_turn = ""
        if len(self.history.get(step, [])) > 0:
            for memory in self.history[step]:
                print(f"Memory: {memory['date']}/{memory['conversation_id']}/{memory['document_type']}: {memory['speaker']}")
                user_turn += f"""- Memory from {memory['date']}: {memory['speaker'].strip()}: {memory['content'].strip()}\n\n"""
            user_turn += """\n"""

        return user_turn

    def format_conscious(self) -> str:
        """
        Formats the "conscious" memories, which are a set of memories that represent the current state of the bot's internal consciousness.
        
        Returns:
            str: A formatted string representing the conscious memories.
        """
                
        conscious = ""
        for memory in self.conscious:
            print(f"Conscious: {memory['date']}/{memory['conversation_id']}")
            conscious += f"""- Personal Journal from {memory['date']}: {memory['content']}\n\n"""
        return conscious

    def format_all(self) -> str:
        """
        Formats the complete conversation history, including the "conscious" memories and the history for each step of the conversation.
        
        Returns:
            str: A formatted string representing the complete conversation history.
        """
                
        history = self.prompt_prefix
        
        history += self.format_conscious()

        for step in list(self.history.keys())[::-1]:
            history += self.format_history(step)
        return history

    def cycle_conscious(self) -> str:
        """
        Cycles the "conscious" memories, which are a set of memories that represent the current state of the bot's internal consciousness.
        
        Returns:
            str: A string representing the formatted "conscious" memories.
        """
                
        self.conscious = [r.to_dict() for _, r in self.cvm.get_conscious(persona_id=self.config.persona_id, top_n=self.config.recall_size).iterrows()]

    def query_memories(self, query_texts: List[str], top_n: int, turn_decay: float = 0.0, temporal_decay: float = 0.8, filter_metadocs: bool = True, **kwargs) -> List[Dict[str, str]]:
        """
        Queries the conversation model for relevant memories based on the provided query texts.
        
        Args:
            query_texts (List[str]): The list of query texts to search for in the conversation model.
            top_n (int): The maximum number of results to return.
            turn_decay (float, optional): The decay factor to apply to memories based on their turn number. Defaults to 0.0.
            temporal_decay (float, optional): The decay factor to apply to memories based on their age. Defaults to 0.8.
            filter_metadocs (bool, optional): Whether to filter out metadata documents. Defaults to True.
        
        Returns:
            List[Dict[str, str]]: A list of dictionaries representing the top N most relevant memories, where each dictionary contains the memory's content and metadata.
        """
                
        print(f"Querying memories for {len(query_texts)} ({top_n})")
        seen_docs = set([r['doc_id'] for h in self.history.values() for r in h])
        cons_docs = set([r['doc_id'] for r in self.conscious])
        filter_docs = seen_docs.union(cons_docs)
        return self.cvm.query(query_texts=query_texts, filter_text=self.filter_text,
                              filter_doc_ids=filter_docs, top_n=top_n,
                              turn_decay=turn_decay, temporal_decay=temporal_decay, filter_metadocs=filter_metadocs)

    def execute_turn(self, step: int, prompt: str, use_guidance: bool = False, max_tokens: int = 512, retry: bool = True, top_n: int = 0, **kwargs) -> str:
        """
        Executes a single turn of the conversation, including:
        - Cycling the "conscious" memories
        - Querying the conversation model for relevant memories based on the user's prompt
        - Accumulating the relevant memories
        - Formatting the complete conversation history
        - Generating a response using the conversation history
        - Optionally retrying the response generation if configured
        
        Args:
            step (int): The current step or turn number in the conversation.
            prompt (str): The user's prompt for the current turn.
            use_guidance (bool, optional): Whether to use any configured guidance when generating the response. Defaults to False.
            max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 512.
            retry (bool, optional): Whether to allow retrying the response generation if it fails. Defaults to True.
            top_n (int, optional): The maximum number of relevant memories to query and accumulate. Defaults to 0 (no memory accumulation).
            **kwargs: Additional keyword arguments to pass to the memory querying and accumulation functions.
        
        Returns:
            str: The generated response for the current turn.
        """
                
        self.cycle_conscious()
        if top_n > 0:
            # This is Active Memory
            texts = [h['content'] for _, s in self.history.items() for h in s if h['role'] == ROLE_ASSISTANT]
            if len(texts) > 0:
                queries = self.query_memories(query_texts=texts, top_n=top_n, **kwargs)
                self.accumulate(step, queries, **kwargs)

        user_turn = prompt
            
        self.turns.append({"role": ROLE_USER, "content": user_turn})
        self.config.max_tokens = max_tokens

        turns = [*self.turns]
        history = self.format_all()
        turns.insert(0, {"role": ROLE_USER, "content": history})
        if use_guidance and self.config.guidance is not None:
            # pop off our user turn and inject our guidance at the beginning
            user_turn = turns.pop()
            content = user_turn['content']
            guidance = self.config.guidance
            new_content = f"{guidance}\n\n{content}"
            turns.append({"role": ROLE_USER, "content": new_content})
        
        response = self.generate_response(turns, self.config)

        if self.config.no_retry == False and retry == True:
            ui = input("** -=[ <enter> or (r)etry ]=- **")
            if ui == 'r':
                raise RetryException()

        return response

    def accept_response(self, response: str, step: int, branch: int, document_type: Optional[str] = None, document_weight: Optional[int] = 1.0, **kwargs):
        """
        Accepts a response generated by the pipeline and appends it to the conversation history.
        
        Args:
            response (str): The generated response to be accepted.
            step (int): The current step or turn number in the conversation.
            branch (int): The current branch or path in the conversation.
            document_type (Optional[str], optional): The type of document to associate with the response. Defaults to None.
            document_weight (Optional[float], optional): The weight or importance of the document. Defaults to 1.0.
            **kwargs: Additional keyword arguments to be passed to the underlying methods.
        
        Returns:
            None
        """
                
        self.turns.append({"role": ROLE_ASSISTANT, "content": response})

        if document_type is not None:
            self.cvm.insert(
                document_type=document_type, conversation_id=self.config.conversation_id, inference_model=self.llm.model,
                user_id=self.config.persona_id, persona_id=self.config.persona_id, sequence_no=step, branch=branch,
                role=ROLE_ASSISTANT, content=response, listener_id=LISTENER_SELF, weight=document_weight)

    @classmethod
    def from_config(cls, config: ChatConfig, persona: Optional[Persona] = None, cvm: Optional[ConversationModel] = None, llm: Optional[LLMProvider] = None) -> 'BasePipeline':
        """
        Constructs a new instance of the `BasePipeline` class from the provided configuration.
        
        Args:
            config (ChatConfig): The configuration settings for the pipeline.
            persona (Optional[Persona], optional): The persona to use for the pipeline. If not provided, the persona will be loaded from the specified persona path.
            cvm (Optional[ConversationModel], optional): The conversation model to use for the pipeline. If not provided, a new `ConversationModel` instance will be created from the configuration.
            llm (Optional[LLMProvider], optional): The language model provider to use for the pipeline. If not provided, a new `LLMProvider` instance will be created from the configuration.
        
        Returns:
            BasePipeline: A new instance of the `BasePipeline` class.
        
        Raises:
            FileNotFoundError: If the specified persona file is not found.
        """

        if persona is None:
            persona_file = os.path.join(config.persona_path, f"{config.persona_id}.json")
            if not os.path.exists(persona_file):
                raise FileNotFoundError(f"Persona {config.persona_id} not found in {config.persona_path}")
            persona = Persona.from_json_file(persona_file)

        if llm is None:
            llm = LLMProvider.from_config(config)

        if cvm is None:
            cvm = ConversationModel.from_uri(**asdict(config))

        return cls(llm, cvm, persona, config)
