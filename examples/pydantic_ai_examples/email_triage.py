"""Email Triage / Classifier — pydantic-ai example.

Classifies incoming emails into priority buckets, extracts intent and action
items, routes them to the right team, and streams a suggested reply draft for
high-priority emails.

Run with::

    ANTHROPIC_API_KEY=your-key python -m pydantic_ai_examples.email_triage

Or as a one-liner with uv::

    ANTHROPIC_API_KEY=your-key \\
      uv run --with "pydantic-ai[examples]" \\
      -m pydantic_ai_examples.email_triage
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

Priority = Literal["critical", "high", "medium", "low", "spam"]
Team = Literal["sales", "support", "billing", "engineering", "hr", "legal", "management", "no-action"]
Intent = Literal[
    "complaint",
    "inquiry",
    "purchase_intent",
    "refund_request",
    "bug_report",
    "feature_request",
    "job_application",
    "partnership",
    "legal_notice",
    "spam",
    "other",
]


class ActionItem(BaseModel):
    description: str
    owner: Team
    due_within_hours: int | None = None


class TriagedEmail(BaseModel):
    subject: str
    sender_name: str | None = None
    sender_email: str | None = None
    priority: Priority
    intent: Intent
    routed_to: Team
    one_line_summary: str = Field(..., description="≤15 words describing the email")
    key_entities: list[str] = Field(
        default_factory=list,
        description="Names, order IDs, product names, etc. mentioned in email",
    )
    action_items: list[ActionItem]
    sentiment: Literal["positive", "neutral", "negative", "urgent"]
    requires_reply: bool
    suggested_reply_tone: Literal["formal", "empathetic", "technical", "brief"] | None = None
    auto_archive: bool = Field(
        ..., description="True for spam or emails needing no action"
    )


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


@dataclass
class TriageDeps:
    company_name: str
    support_email: str
    teams: list[Team] = field(default_factory=lambda: list(Team.__args__))  # type: ignore[attr-defined]
    sla_hours: dict[Priority, int] = field(
        default_factory=lambda: {
            "critical": 1,
            "high": 4,
            "medium": 24,
            "low": 72,
            "spam": 0,
        }
    )


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

triage_agent: Agent[TriageDeps, TriagedEmail] = Agent(
    "anthropic:claude-sonnet-4-20250514",
    deps_type=TriageDeps,
    output_type=TriagedEmail,
    system_prompt=(
        "You are an intelligent email triage system. "
        "Analyze the email provided and classify it precisely. "
        "Extract all action items with realistic due times based on priority. "
        "If the email looks like spam or a mass marketing message, mark it accordingly."
    ),
)


@triage_agent.system_prompt
async def company_context(ctx: RunContext[TriageDeps]) -> str:
    sla_text = ", ".join(
        f"{p}={h}h" for p, h in ctx.deps.sla_hours.items()
    )
    return (
        f"Company: {ctx.deps.company_name}\n"
        f"Support email: {ctx.deps.support_email}\n"
        f"SLA targets: {sla_text}"
    )


reply_agent: Agent[TriageDeps, str] = Agent(
    "anthropic:claude-sonnet-4-20250514",
    deps_type=TriageDeps,
    output_type=str,
    system_prompt=(
        "You are a professional customer communications specialist. "
        "Write a helpful, appropriately-toned reply to the given email. "
        "Keep it concise (under 150 words unless absolutely necessary). "
        "Do not make promises you can't keep. Use a professional closing."
    ),
)


@reply_agent.system_prompt
async def reply_company_context(ctx: RunContext[TriageDeps]) -> str:
    return f"You are writing on behalf of {ctx.deps.company_name}."


# ---------------------------------------------------------------------------
# Sample emails
# ---------------------------------------------------------------------------

SAMPLE_EMAILS = [
    {
        "from": "Ahmad Tariq <ahmad.tariq@bigclient.com>",
        "subject": "URGENT: Production system down — Order #98712",
        "body": """
        Hi Team,

        Our entire checkout flow has been broken since 9 AM PST. We are losing
        approximately $10,000/hour in revenue. Order #98712 for 500 enterprise
        licenses is at risk of cancellation if this isn't resolved in the next
        2 hours.

        This is completely unacceptable. Please escalate immediately.

        Ahmad Tariq
        VP Operations, BigClient Inc.
        """,
    },
    {
        "from": "newsletter@randomstore.com",
        "subject": "🔥 SALE! 50% off everything this weekend only!!!",
        "body": "Click here to shop now! Limited time offer blah blah...",
    },
    {
        "from": "Sara Khan <sara.k@freelancer.com>",
        "subject": "Question about your Enterprise plan pricing",
        "body": """
        Hello,

        I came across your platform and I'm interested in learning more about
        the Enterprise plan. Could you share pricing details and whether you
        offer a trial period? We're a team of about 50 people.

        Thanks,
        Sara
        """,
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    deps = TriageDeps(
        company_name="TechPlatform Inc.",
        support_email="support@techplatform.com",
    )

    for email in SAMPLE_EMAILS:
        print("=" * 60)
        print(f"Subject: {email['subject']}")
        print(f"From   : {email['from']}")
        print("=" * 60)

        email_text = (
            f"From: {email['from']}\n"
            f"Subject: {email['subject']}\n\n"
            f"{email['body']}"
        )

        result = await triage_agent.run(email_text, deps=deps)
        triaged = result.output

        print(f"Priority      : {triaged.priority.upper()}")
        print(f"Intent        : {triaged.intent}")
        print(f"Route to      : {triaged.routed_to}")
        print(f"Sentiment     : {triaged.sentiment}")
        print(f"Summary       : {triaged.one_line_summary}")
        print(f"Auto-archive  : {triaged.auto_archive}")
        print(f"Needs reply   : {triaged.requires_reply}")

        if triaged.action_items:
            print("Action Items:")
            for item in triaged.action_items:
                due = f" (due in {item.due_within_hours}h)" if item.due_within_hours else ""
                print(f"  → [{item.owner}]{due} {item.description}")

        if triaged.requires_reply and not triaged.auto_archive:
            print(f"\nStreaming reply draft (tone: {triaged.suggested_reply_tone})...\n")
            async with reply_agent.run_stream(
                f"Original email:\n{email_text}\n\n"
                f"Triage context:\n{triaged.model_dump_json(indent=2)}\n\n"
                "Write the reply now.",
                deps=deps,
            ) as stream:
                async for chunk in stream.stream_text(delta=True):
                    print(chunk, end="", flush=True)

        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
