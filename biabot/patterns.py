# biabot/patterns.py

import re
from typing import Optional

from .config import ChatConfig


class Patterns:
    def __init__(self, config: ChatConfig):
        self.config = config
        
        self.patterns = {
            "total_steps": re.compile(r"(?:Total\s+Steps?|Steps?\s+Total):\s*(\d+)", re.IGNORECASE)
        }

    def extract_total_steps(self, response: str) -> Optional[str]:
        """
        Extracts the total number of steps from the given response string, looking for the words "total" or "steps" followed by a colon and a number.
        
        Args:
            response (str): The response string to search for the total steps.
        
        Returns:
            Optional[str]: The total number of steps as a string, or None if not found.
        """
                
        matches = self.patterns['total_steps'].findall(response)
        if matches:
            for match in matches:
                try:
                    return int(match)
                except ValueError:
                    pass
        return None