"""Numeric command-line values shared by the launcher and trainer."""

from fractions import Fraction


def parse_float(value: str) -> float:
    """Accept decimals, scientific notation, or fractions such as 96/384."""
    try:
        return float(value)
    except ValueError:
        try:
            return float(Fraction(value))
        except (ValueError, ZeroDivisionError, OverflowError) as error:
            raise ValueError(f"expected a number or fraction, got {value!r}") from error
