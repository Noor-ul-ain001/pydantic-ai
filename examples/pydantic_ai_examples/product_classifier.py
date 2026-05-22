"""Ecommerce Product Classifier — pydantic-ai example.

Classifies raw product listings into a structured catalog schema, assigns
categories and tags, detects quality issues, and streams a merchandising
description ready for a product page.

Run with::

    ANTHROPIC_API_KEY=your-key python -m pydantic_ai_examples.product_classifier

Or as a one-liner with uv::

    ANTHROPIC_API_KEY=your-key \\
      uv run --with "pydantic-ai[examples]" \\
      -m pydantic_ai_examples.product_classifier
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

Category = Literal[
    "Electronics",
    "Clothing & Apparel",
    "Home & Kitchen",
    "Sports & Outdoors",
    "Books & Media",
    "Health & Beauty",
    "Toys & Games",
    "Automotive",
    "Other",
]


class ProductAttribute(BaseModel):
    name: str   # e.g. "Color", "Material", "Weight"
    value: str  # e.g. "Midnight Blue", "Cotton", "1.2 kg"


class QualityIssue(BaseModel):
    field: str
    issue: str
    severity: Literal["minor", "major", "blocking"]


class ClassifiedProduct(BaseModel):
    product_id: str
    title: str = Field(..., description="Clean, normalized product title")
    category: Category
    subcategory: str
    brand: str | None = None
    price_usd: float | None = None
    currency_detected: str | None = None
    attributes: list[ProductAttribute]
    tags: list[str] = Field(..., max_length=15, description="SEO and filter tags")
    target_audience: list[Literal["men", "women", "children", "unisex", "professional"]]
    quality_issues: list[QualityIssue] = Field(default_factory=list)
    is_publishable: bool = Field(
        ..., description="True only if no blocking quality issues"
    )


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


@dataclass
class ClassifierDeps:
    store_name: str
    allowed_categories: list[Category] = field(
        default_factory=lambda: list(Category.__args__)  # type: ignore[attr-defined]
    )
    min_title_length: int = 10
    require_brand: bool = False


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

classifier_agent: Agent[ClassifierDeps, ClassifiedProduct] = Agent(
    "anthropic:claude-sonnet-4-20250514",
    deps_type=ClassifierDeps,
    output_type=ClassifiedProduct,
    system_prompt=(
        "You are an ecommerce catalog specialist. "
        "Given a raw product listing, extract and normalize all product data "
        "into the structured schema. Identify any quality issues that would "
        "prevent the listing from being published."
    ),
)


@classifier_agent.system_prompt
async def store_context(ctx: RunContext[ClassifierDeps]) -> str:
    cats = ", ".join(ctx.deps.allowed_categories)
    rules = [f"Store: {ctx.deps.store_name}"]
    rules.append(f"Allowed categories: {cats}")
    rules.append(f"Min title length: {ctx.deps.min_title_length} chars")
    if ctx.deps.require_brand:
        rules.append("Brand is required — flag as blocking if missing.")
    return "\n".join(rules)


description_agent: Agent[None, str] = Agent(
    "anthropic:claude-sonnet-4-20250514",
    output_type=str,
    system_prompt=(
        "You are a copywriter for an ecommerce store. "
        "Write a compelling, SEO-friendly product description (100-150 words) "
        "based on the classified product data provided. "
        "Use bullet points for key features. Keep the tone engaging and professional."
    ),
)


# ---------------------------------------------------------------------------
# Sample raw product listings
# ---------------------------------------------------------------------------

SAMPLE_LISTINGS = [
    {
        "product_id": "P001",
        "raw_text": """
        nike air max 270 mens running shoes size 10 black
        price: $120 USD
        material: mesh upper, rubber sole
        wt: 310g
        """,
    },
    {
        "product_id": "P002",
        "raw_text": """
        Instant Pot Duo 7-in-1 Electric Pressure Cooker, 6 Quart
        Brand: Instant Pot
        Price: PKR 25,000
        Features: pressure cooker, slow cooker, rice cooker, steamer,
        sauté, yogurt maker, warmer
        Color: Stainless Steel
        """,
    },
    {
        "product_id": "P003",
        "raw_text": "good product buy now sale!!",  # intentionally low quality
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    deps = ClassifierDeps(store_name="ShopGlobal", require_brand=False)

    for listing in SAMPLE_LISTINGS:
        print("=" * 60)
        print(f"Product ID: {listing['product_id']}")
        print("=" * 60)

        result = await classifier_agent.run(
            f"Classify this product listing:\n\n{listing['raw_text']}",
            deps=deps,
        )
        product = result.output

        print(f"Title         : {product.title}")
        print(f"Category      : {product.category} > {product.subcategory}")
        print(f"Brand         : {product.brand or 'N/A'}")
        print(f"Price (USD)   : {product.price_usd or 'N/A'}")
        print(f"Tags          : {', '.join(product.tags[:6])}")
        print(f"Publishable   : {'YES' if product.is_publishable else 'NO'}")

        if product.quality_issues:
            print("Quality Issues:")
            for issue in product.quality_issues:
                print(f"  [{issue.severity.upper():8s}] {issue.field}: {issue.issue}")

        if product.is_publishable:
            print("\nStreaming product description...\n")
            async with description_agent.run_stream(
                f"Write a product description for:\n{product.model_dump_json(indent=2)}"
            ) as stream:
                async for chunk in stream.stream_text(delta=True):
                    print(chunk, end="", flush=True)
            print()

        print()


if __name__ == "__main__":
    asyncio.run(main())
