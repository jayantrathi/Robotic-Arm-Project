#!/usr/bin/env python3
"""
The objects the arm knows, in one place.

Each object maps a canonical name to:
  - detect: the open-vocab label(s) OWLv2 responds to best (what the DETECTOR
    looks for -- "teddy bear" localizes better than "bear")
  - task: the exact instruction string the SmolVLA policy was trained on (what
    the POLICY is given to drive the grasp)
  - aliases: words a person might say/type for this object (for voice + typed
    command parsing)

Adding a new object is a zero-retraining change for the DETECTOR (just name it);
the grasp policy still has to have been trained on a similar object for the
grasp itself to work.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KnownObject:
    name: str                       # canonical key, e.g. "pen"
    detect: list[str]               # OWLv2 label(s) for localization
    task: str                       # trained instruction for the policy
    aliases: list[str] = field(default_factory=list)


REGISTRY: dict[str, KnownObject] = {
    "pen": KnownObject(
        name="pen",
        detect=["pen"],
        task="Pick up the pen and drop it on the plate",
        aliases=["pen", "marker", "sharpie"],
    ),
    "bear": KnownObject(
        name="bear",
        detect=["teddy bear"],
        task="Pick up the bear and drop it on the plate",
        aliases=["bear", "teddy", "teddy bear", "toy"],
    ),
    "lip balm": KnownObject(
        name="lip balm",
        detect=["lip balm"],
        task="Pick up the lip balm and drop it on the plate",
        aliases=["lip balm", "balm", "chapstick", "tube"],
    ),
}


def all_detect_labels() -> list[str]:
    """Every detector label across all known objects (de-duplicated, ordered).

    Handing the full set to the detector as candidates makes it score the
    objects against each other, which is far more decisive than asking about
    one label in isolation.
    """
    labels: list[str] = []
    for obj in REGISTRY.values():
        for d in obj.detect:
            if d not in labels:
                labels.append(d)
    return labels


def resolve(spoken_or_typed: str) -> KnownObject | None:
    """Map a free-text command ('grab the teddy', 'get me the pen') to an object.

    Matches on whole-word-ish substring against each object's aliases, longest
    alias first so 'lip balm' wins over a stray 'b'.
    """
    text = spoken_or_typed.lower()
    best: tuple[int, KnownObject] | None = None
    for obj in REGISTRY.values():
        for alias in obj.aliases:
            if alias in text:
                if best is None or len(alias) > best[0]:
                    best = (len(alias), obj)
    return best[1] if best else None
