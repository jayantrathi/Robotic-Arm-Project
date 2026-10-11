from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KnownObject:
    name: str
    detect: list[str]
    task: str
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
    "headphone case": KnownObject(
        name="headphone case",
        detect=["pouch"],
        task="Pick up the headphone case and drop it on the plate",
        aliases=["headphone case", "headphones", "headphone", "case", "pouch"],
    ),
}


def all_detect_labels() -> list[str]:
    labels: list[str] = []
    for obj in REGISTRY.values():
        for d in obj.detect:
            if d not in labels:
                labels.append(d)
    return labels


def resolve(spoken_or_typed: str) -> KnownObject | None:
    text = spoken_or_typed.lower()
    best: tuple[int, KnownObject] | None = None
    for obj in REGISTRY.values():
        for alias in obj.aliases:
            if alias in text:
                if best is None or len(alias) > best[0]:
                    best = (len(alias), obj)
    return best[1] if best else None
