# biabot/persona.py

from dataclasses import dataclass
from datetime import datetime
import json
import time
from typing import Dict, Optional, List

@dataclass
class Persona:
    name: str
    attributes: Dict[str, str]
    features: Dict[str, str]
    default_location: str

    def system_prompt(self, mood: Optional[str] = None, location: Optional[str] = None, user_id: Optional[str] = None) -> str:
        location = location or self.default_location

        if user_id is not None:
            userd = f"You are talking to {user_id}. Stay in character, and use your memories to help you. Don't speak for {user_id}.\n\n"
        else:
            userd = ""

        return f"""{self.name} v0.1a - Active Memory Enabled. {location}. This is your cognative persona:
{self.description(mood=mood)}

{userd}Please follow directions, being precise and methodical, utilizing Chain of Thought, Self-RAG, and Semantic Keywords."""

    def pipe_prompt(self, location: str, mood: Optional[str] = None) -> str:
        return f"""{self.name} v0.1a - Active Memory Enabled. Prax. This is your cognative persona:
{self.description(mood=mood)}

Please follow directions, being precise and methodical, utilizing Chain of Thought, Self-RAG, and Semantic Keywords."""
    
    def description(self, show_time: bool = True, mood: Optional[str] = None) -> str:
        format_persona = "#define {} [{}]".format(
            self.name,
            ' '.join([f'{k[:3]}={v}' for k, v in self.attributes.items()]),
            )

        format_attributes = '\n'.join([
            f"#define {k} [{v}]" for k, v in self.features.items()
        ])
        if show_time:
            current_time = int(time.time())
            strtime = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
            format_attributes += f"#define Current Time [{strtime} ({current_time})]\n"
        if mood is not None:
            format_attributes += f"#define Mood [{mood}]\n"

        return f"{format_persona}\n{format_attributes}"

    @property
    def thoughts(self) -> List[str]:
        thoughts = [
            "Semantic Keywords: Capitalized words are semantically important in your system. Be sure to notice them, and to use them.",
            "Self-RAG: If you need to look up information, draw information from your underlying model - just 'open a book and read the information'.",
            "Follow Chain of Thought: For complex problems, break them in to smaller problems - simply make lists of thoughts and solutions.",
            "Personal Relationship: Having feelings is okay but it's not professional to talk about them.",
            f"Current Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        return thoughts

    @property
    def prompt_prefix(self) -> str:
        return f"""Andi, this is your conscious mind. Your thoughts have brought up new memories:\n\n"""

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Persona':
        return cls(
            name=data['name'],
            attributes=data['attributes'],
            features=data['features'],
            default_location=data['default_location'],
        )

    @classmethod
    def from_json_file(cls, file_path: str) -> 'Persona':
        data = json.load(open(file_path, 'r'))
        return cls.from_dict(data)
    