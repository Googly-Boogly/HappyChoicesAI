from dataclasses import dataclass, field
from typing import Dict, List, Tuple


def input_situation() -> str:
    """
    Captures and returns the situation description provided by the user.
    """
    return input("Describe the situation: ")
