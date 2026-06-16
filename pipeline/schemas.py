"""schemas.py — Structured-output schemas (one per answer kind) + helpers.

Forcing the LLM into these shapes is how we get exact typed answers, and the
Optional/empty cases are how the model signals "abstain".
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---- answer schemas, keyed by question `kind` ----
class NumberAnswer(BaseModel):
    """A single numeric answer extracted from the report."""
    value: Optional[float] = Field(
        None, description="The number only (no units/commas). null if not in the context.")


class BooleanAnswer(BaseModel):
    value: bool = Field(
        description="True only if the context clearly supports it; otherwise False.")


class NameAnswer(BaseModel):
    value: Optional[str] = Field(
        None, description="The single name/title. null if not in the context.")


class NamesAnswer(BaseModel):
    values: List[str] = Field(
        default_factory=list, description="List of names/titles; empty if none found.")


_SCHEMAS = {"number": NumberAnswer, "boolean": BooleanAnswer,
            "name": NameAnswer, "names": NamesAnswer}


# ---- control schemas ----
class Grade(BaseModel):
    sufficient: bool = Field(description="Do these chunks contain enough to answer the question?")


class Grounded(BaseModel):
    grounded: bool = Field(description="Is the answer fully supported by the context?")


class Decomposition(BaseModel):
    metric: str = Field(description="The financial metric being compared, e.g. 'total assets'.")
    direction: Literal["lowest", "highest"]


def schema_for_kind(kind: str):
    return _SCHEMAS.get(kind, NameAnswer)


def default_for_kind(kind: str):
    """The abstention value per kind."""
    return False if kind == "boolean" else "N/A"


def finalize_answer(kind: str, answer) -> object:
    """Convert a pydantic answer object into the submission value."""
    if kind == "number":
        return answer.value if answer.value is not None else "N/A"
    if kind == "boolean":
        return bool(answer.value)
    if kind == "name":
        return answer.value if answer.value else "N/A"
    if kind == "names":
        return answer.values if answer.values else "N/A"
    return "N/A"
