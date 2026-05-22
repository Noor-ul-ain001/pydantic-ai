"""Code Review Agent — pydantic-ai example.

Performs a structured code review: detects bugs, security issues, style
violations and performance problems, then streams a detailed review comment
in the style of a GitHub PR review.

Run with::

    ANTHROPIC_API_KEY=your-key python -m pydantic_ai_examples.code_review

Or as a one-liner with uv::

    ANTHROPIC_API_KEY=your-key \\
      uv run --with "pydantic-ai[examples]" \\
      -m pydantic_ai_examples.code_review
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

Severity = Literal["nitpick", "suggestion", "warning", "error", "critical"]
Category = Literal[
    "bug",
    "security",
    "performance",
    "readability",
    "maintainability",
    "type_safety",
    "test_coverage",
    "documentation",
    "style",
]


class ReviewComment(BaseModel):
    line_range: str | None = Field(
        None, description="e.g. 'L12', 'L24-L31', or None for file-level comments"
    )
    category: Category
    severity: Severity
    title: str
    explanation: str
    suggested_fix: str | None = None


class CodeReview(BaseModel):
    language: str
    file_path: str | None = None
    overall_verdict: Literal["approve", "approve_with_suggestions", "request_changes", "block"]
    quality_score: int = Field(..., ge=0, le=10, description="0 = unacceptable, 10 = exemplary")
    complexity_assessment: Literal["simple", "moderate", "complex", "over-engineered"]
    test_coverage_estimate: Literal["none", "low", "adequate", "good", "excellent"] | None = None
    comments: list[ReviewComment]
    positive_aspects: list[str] = Field(
        default_factory=list,
        description="Things done well — always include at least one if code is non-trivial",
    )
    summary: str = Field(..., description="2-3 sentence overall assessment")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


@dataclass
class ReviewDeps:
    team_standards: list[str] = field(
        default_factory=lambda: [
            "Use type hints for all function signatures",
            "No bare except clauses",
            "All SQL queries must use parameterized inputs",
            "Functions should not exceed 50 lines",
            "Secrets must never be hardcoded",
        ]
    )
    language: str = "Python"
    strict_mode: bool = False   # if True, treat warnings as errors


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

review_agent: Agent[ReviewDeps, CodeReview] = Agent(
    "anthropic:claude-sonnet-4-20250514",
    deps_type=ReviewDeps,
    output_type=CodeReview,
    system_prompt=(
        "You are a senior software engineer conducting a thorough code review. "
        "Be specific — reference actual line numbers or code snippets in your comments. "
        "Prioritize correctness and security over style. "
        "Be constructive, not harsh — always explain WHY something is an issue."
    ),
)


@review_agent.system_prompt
async def inject_standards(ctx: RunContext[ReviewDeps]) -> str:
    standards = "\n".join(f"  - {s}" for s in ctx.deps.team_standards)
    mode = "STRICT: treat all warnings as errors." if ctx.deps.strict_mode else "Standard review mode."
    return (
        f"Language: {ctx.deps.language}\n"
        f"Mode: {mode}\n"
        f"Team standards to enforce:\n{standards}"
    )


pr_comment_agent: Agent[None, str] = Agent(
    "anthropic:claude-sonnet-4-20250514",
    output_type=str,
    system_prompt=(
        "You are a senior engineer writing a GitHub PR review comment. "
        "Format your response as a proper Markdown PR review. "
        "Use headers, code blocks, and emoji sparingly but effectively. "
        "Start with the verdict and score, then group comments by severity. "
        "End with a summary and clear next steps for the author."
    ),
)


# ---------------------------------------------------------------------------
# Sample code snippets to review
# ---------------------------------------------------------------------------

SAMPLE_CODE = '''
# file: auth/user_service.py

import sqlite3
import hashlib

DB_PASSWORD = "super_secret_123"  # production password
ADMIN_TOKEN = "Bearer abc123xyz"

def get_user(username):
    conn = sqlite3.connect("users.db", password=DB_PASSWORD)
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()

def authenticate(username, password):
    user = get_user(username)
    if user:
        stored_hash = user[2]
        input_hash = hashlib.md5(password.encode()).hexdigest()
        if stored_hash == input_hash:
            return True
    return False

def create_user(username, password, role="user"):
    conn = sqlite3.connect("users.db", password=DB_PASSWORD)
    cursor = conn.cursor()
    password_hash = hashlib.md5(password.encode()).hexdigest()
    try:
        cursor.execute(
            f"INSERT INTO users VALUES ('{username}', '{role}', '{password_hash}')"
        )
        conn.commit()
    except:
        pass  # silently ignore errors

def delete_user(username):
    conn = sqlite3.connect("users.db", password=DB_PASSWORD)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM users WHERE username = '{username}'")
    conn.commit()
'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    deps = ReviewDeps(
        language="Python",
        strict_mode=True,
    )

    print("=" * 60)
    print("STEP 1 — Running structured code analysis...")
    print("=" * 60)

    result = await review_agent.run(
        f"Review this code:\n\n```python\n{SAMPLE_CODE}\n```",
        deps=deps,
    )
    review = result.output

    print(f"\nLanguage      : {review.language}")
    print(f"Verdict       : {review.overall_verdict.upper().replace('_', ' ')}")
    print(f"Quality Score : {review.quality_score}/10")
    print(f"Complexity    : {review.complexity_assessment}")
    print(f"Total Issues  : {len(review.comments)}")

    severity_order = ["critical", "error", "warning", "suggestion", "nitpick"]
    for sev in severity_order:
        issues = [c for c in review.comments if c.severity == sev]
        if issues:
            print(f"\n  {sev.upper()} ({len(issues)}):")
            for issue in issues:
                loc = f" {issue.line_range}" if issue.line_range else ""
                print(f"    [{issue.category}]{loc} — {issue.title}")

    if review.positive_aspects:
        print(f"\nPositives     :")
        for pos in review.positive_aspects:
            print(f"  + {pos}")

    print("\n" + "=" * 60)
    print("STEP 2 — Streaming formatted PR review comment...")
    print("=" * 60 + "\n")

    async with pr_comment_agent.run_stream(
        f"Write a GitHub PR review comment based on this analysis:\n"
        f"{review.model_dump_json(indent=2)}\n\n"
        f"Original code:\n```python\n{SAMPLE_CODE}\n```"
    ) as stream:
        async for chunk in stream.stream_text(delta=True):
            print(chunk, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
