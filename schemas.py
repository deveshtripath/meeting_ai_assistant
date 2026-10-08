"""Pydantic schema for the structured meeting record + renderers (Markdown / JSON)."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator

UNSPECIFIED = "unspecified"


class MinuteSection(BaseModel):
    topic: str
    points: List[str] = Field(default_factory=list)


class Decision(BaseModel):
    decision: str
    context: str = ""


class ActionItem(BaseModel):
    task: str
    owner: str = UNSPECIFIED
    deadline: str = UNSPECIFIED

    @field_validator("owner", "deadline", mode="before")
    @classmethod
    def _normalise_missing(cls, v):
        if v is None:
            return UNSPECIFIED
        s = str(v).strip()
        if s.lower() in {"", "none", "n/a", "na", "null", "not specified", "unknown", "tbd", "unspecified"}:
            return UNSPECIFIED
        return s


class MeetingRecord(BaseModel):
    summary: str
    minutes: List[MinuteSection] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)


def to_markdown(record: MeetingRecord, title: str = "Meeting Record") -> str:
    """Human-readable rendering generated from the SAME object as the JSON output."""
    lines = [f"# {title}", "", "## Summary", "", record.summary, "", "## Minutes", ""]
    if record.minutes:
        for sec in record.minutes:
            lines.append(f"### {sec.topic}")
            lines += [f"- {p}" for p in sec.points]
            lines.append("")
    else:
        lines += ["_No minutes generated._", ""]

    lines += ["## Key Decisions", ""]
    if record.decisions:
        for i, d in enumerate(record.decisions, 1):
            ctx = f" — {d.context}" if d.context else ""
            lines.append(f"{i}. {d.decision}{ctx}")
    else:
        lines.append("_No decisions were made in this meeting._")
    lines.append("")

    lines += ["## Action Items", ""]
    if record.action_items:
        lines += ["| # | Task | Owner | Deadline |", "|---|------|-------|----------|"]
        for i, a in enumerate(record.action_items, 1):
            lines.append(f"| {i} | {a.task} | {a.owner} | {a.deadline} |")
    else:
        lines.append("_No action items were assigned in this meeting._")
    lines.append("")
    return "\n".join(lines)
