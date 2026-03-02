from dataclasses import dataclass

@dataclass(frozen=True)
class Zone:
    """Simple rectangular zone (pixel coords)."""
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2