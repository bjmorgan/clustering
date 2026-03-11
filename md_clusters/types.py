"""Bond specification data type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BondSpec:
    """Declarative rule for bond detection between a species pair.

    The species pair is stored in sorted order so that the rule is
    invariant under exchange of the two labels.  Species names support
    fnmatch-style wildcards (``*``, ``?``).

    Attributes:
        species: Sorted pair of species patterns.
        max_length: Maximum bond length threshold (angstroms).
        min_length: Minimum bond length threshold (angstroms).
    """

    species: tuple[str, str]
    max_length: float
    min_length: float = 0.0

    def __post_init__(self) -> None:
        # Sort the species pair for order-invariance.
        a, b = sorted(self.species)
        object.__setattr__(self, "species", (a, b))

        if self.max_length <= 0:
            raise ValueError(
                f"max_length must be positive, got {self.max_length}"
            )
        if self.min_length < 0:
            raise ValueError(
                f"min_length must be non-negative, got {self.min_length}"
            )
        if self.min_length > self.max_length:
            raise ValueError(
                f"min_length ({self.min_length}) must not exceed "
                f"max_length ({self.max_length})"
            )
