"""Reusable prompt template helpers."""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from textwrap import dedent
from typing import Any, Mapping


@dataclass(slots=True)
class RenderedPrompt:
    """Container with fully rendered system/user instructions."""

    system: str
    user: str


@dataclass(slots=True)
class PromptTemplate:
    """Simple string.Template powered prompt definition."""

    name: str
    system: str
    user: str

    def render(self, context: Mapping[str, Any]) -> RenderedPrompt:
        """Render template with provided context."""
        system_template = Template(dedent(self.system).strip())
        user_template = Template(dedent(self.user).strip())
        return RenderedPrompt(
            system=system_template.safe_substitute(**context),
            user=user_template.safe_substitute(**context),
        )

