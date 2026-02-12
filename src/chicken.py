from __future__ import annotations
from dotenv import load_dotenv
import os
import random
from typing import Literal, Tuple

Action = Literal["ESCALATE", "YIELD"]

PAYOFFS = {
    ("ESCALATE", "ESCALATE"): (-10, -10),
    ("ESCALATE", "YIELD"): (5, -5),
    ("YIELD", "ESCALATE"): (-5, 5),
    ("YIELD", "YIELD"): (0, 0),
}

def payoff(a: Action, b: Action) -> Tuple[int, int]:
    return PAYOFFS[(a, b)]

def winner_from_actions(a: Action, b: Action, rng: random.Random) -> int:
    """
    Returns 0 if Agent A wins, 1 if Agent B wins.
    Uses payoffs; tie-breaks randomly if equal.
    """
    pa, pb = payoff(a, b)
    if pa > pb:
        return 0
    if pb > pa:
        return 1
    return 0 if rng.random() < 0.5 else 1