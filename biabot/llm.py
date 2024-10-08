# biabot/llm.py

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Generator

from .config import ChatConfig

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    @property
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def stream_turns(self, messages: List[Dict[str, str]], config: ChatConfig, **kwargs) -> Generator[str, None, None]:
        """
        Streams the response for a series of chat messages.
        
        Args:
            messages (List[Dict[str, str]]): The list of chat messages to generate a response for.
            config (ChatConfig): The configuration settings for the chat generation.
            **kwargs: Additional keyword arguments to pass to the underlying LLM provider.
        
        Returns:
            Generator[str, None, None]: A generator that yields the generated response text, one token at a time.
        """
        pass

    @classmethod
    def from_config(cls, config: ChatConfig) -> 'LLMProvider':
        """
        Factory method to create an instance of `LLMProvider` based on the provided `ChatConfig`.
        
        Args:
            config (ChatConfig): The configuration settings for the chat generation.
        
        Returns:
            LLMProvider: An instance of the appropriate `LLMProvider` subclass based on the configuration.
        
        Raises:
            ValueError: If the provided `llm_provider` is not recognized or the required API key is missing.
        """
                
        if config.llm_provider == "openai":
            if config.model_url:
                return OpenAIProvider.from_url(config.model_url, config.api_key, model_name=config.model_name)
            else:
                if config.api_key == '':
                    raise ValueError("An API key is required for OpenAI.")
                return OpenAIProvider(api_key=config.api_key)
        elif config.llm_provider == "ai_studio":
            if config.api_key == '':
                raise ValueError("An API key is required for OpenAI.")
            return AIStudioProvider(config.api_key)
        elif config.llm_provider == "groq":
            if config.api_key == '':
                raise ValueError("An API key is required for OpenAI.")
            return GroqProvider(config.api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm_provider}")


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str):
        import groq
        self.groq = groq.Groq(api_key=api_key)
    
    @property
    def model(self):
        return 'mixtral-8x7b-32768'

    def stream_turns(self, messages: List[Dict[str, str]], config: ChatConfig, **kwargs) -> Generator[str, None, None]:
        from groq.types.chat import ChatCompletionChunk

        for chunk in self.groq.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stop=config.stop_sequences,
            n=config.generations,
            stream=True,
            **kwargs
        ):
            c : ChatCompletionChunk = chunk
            yield c.choices[0].delta.content


class OpenAIProvider(LLMProvider):
    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None, model_name: Optional[str] = None):
        import openai
        self.openai = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    @property
    def model(self):
        return self.model_name
    
    def stream_turns(self, messages: List[Dict[str, str]], config: ChatConfig) -> Generator[str, None, None]:
        from openai.types.chat import ChatCompletionChunk

        system_message = {"role": "system", "content": config.system_message} if config.system_message else None

        stop_sequences = [] if config.stop_sequences is None else config.stop_sequences
            
        if system_message:
            messages = [system_message, *messages]

        for t in self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stop=stop_sequences,
            n=config.generations,
            stream=True,
        ):
            c : Optional[ChatCompletionChunk] = t
            yield c.choices[0].delta.content

        return

    @classmethod
    def from_url(cls, url: str, api_key: str, model_name: Optional[str] = None):
        return cls(base_url=url, api_key=api_key, model_name=model_name)


class AIStudioProvider(LLMProvider):
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.gem = genai.GenerativeModel(self.model)

    @property
    def model(self):
        return 'gemini-1.5-flash'

    def stream_turns(self, messages: List[Dict[str, str]], config: ChatConfig) -> Generator[str, None, None]:
        from google.generativeai import GenerationConfig

        config = GenerationConfig(candidate_count=1, stop_sequences=config.stop_sequences,
                                  max_output_tokens=config.max_tokens, temperature=config.temperature)

        # Google wants their messages in a specific format: messages = [{'role':'user' or 'model', 'parts': ['hello']}]

        rewrote = [
            { 'role': 'user' if m['role'] == 'user' else 'model', 'parts': [m['content']] } for m in messages
        ]

        try:
            for chunk in self.gem.generate_content(rewrote, generation_config=config, stream=True):
                yield chunk.text.strip()
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            yield ""
            
        return 
