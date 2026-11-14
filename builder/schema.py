"""The website schema: the contract between the agent workflow and the renderer.

The frontend only knows how to render what this file allows, and the agents are
only allowed to produce what this file validates. Unknown fields are rejected so
a hallucinated property becomes a validation error (and a repair attempt)
instead of a silently broken preview.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Optional, get_args

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

SectionType = Literal["hero", "text", "image", "features", "testimonials", "pricing", "cta", "contact"]
SECTION_TYPES = get_args(SectionType)

_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _hex(value: str) -> str:
    value = value.strip()
    if not _HEX.match(value):
        raise ValueError("must be a hex colour like #1a2b3c")
    if len(value) == 4:
        value = "#" + "".join(c * 2 for c in value[1:])
    return value.lower()


HexColor = Annotated[str, AfterValidator(_hex)]
Short = Annotated[str, Field(min_length=1, max_length=120)]
Long = Annotated[str, Field(min_length=1, max_length=800)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- per-section props -------------------------------------------------------
class HeroProps(Strict):
    headline: Short
    subheadline: Long = "A short supporting line."
    cta_label: Short = "Get started"
    cta_href: Short = "#contact"


class TextProps(Strict):
    heading: Short
    body: Long


class ImageProps(Strict):
    url: Optional[str] = Field(default=None, max_length=500)  # None -> placeholder block
    alt: Short = "Image"
    caption: Optional[Short] = None


class FeatureItem(Strict):
    title: Short
    description: Long


class FeaturesProps(Strict):
    heading: Short
    items: list[FeatureItem] = Field(min_length=1, max_length=8)


class Testimonial(Strict):
    quote: Long
    author: Short
    role: Optional[Short] = None


class TestimonialsProps(Strict):
    heading: Short
    items: list[Testimonial] = Field(min_length=1, max_length=6)


class PricingPlan(Strict):
    name: Short
    price: Short
    period: Short = "month"
    features: list[Short] = Field(default_factory=list, max_length=8)
    highlighted: bool = False


class PricingProps(Strict):
    heading: Short
    plans: list[PricingPlan] = Field(min_length=1, max_length=4)


class CTAProps(Strict):
    heading: Short
    body: Optional[Long] = None
    button_label: Short = "Get started"
    button_href: Short = "#contact"


class ContactProps(Strict):
    heading: Short
    body: Optional[Long] = None
    email: Optional[Short] = None
    fields: list[Literal["name", "email", "phone", "message"]] = Field(
        default_factory=lambda: ["name", "email", "message"], min_length=1, max_length=4)
    submit_label: Short = "Send"


PROPS_MODELS: dict[str, type[Strict]] = {
    "hero": HeroProps, "text": TextProps, "image": ImageProps, "features": FeaturesProps,
    "testimonials": TestimonialsProps, "pricing": PricingProps, "cta": CTAProps, "contact": ContactProps,
}
assert set(PROPS_MODELS) == set(SECTION_TYPES), "every section type needs exactly one props model"


# ---- section / site ----------------------------------------------------------
class Style(Strict):
    background: Optional[HexColor] = None   # None -> inherit theme background
    text_color: Optional[HexColor] = None   # None -> inherit theme text
    padding: Literal["sm", "md", "lg"] = "md"
    align: Literal["left", "center"] = "center"


class Section(Strict):
    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{0,39}$")]
    type: SectionType
    props: dict[str, Any]
    style: Style = Field(default_factory=Style)

    @model_validator(mode="after")
    def _validate_props(self) -> "Section":
        # Validate props against the model for this section type and normalise them.
        self.props = PROPS_MODELS[self.type].model_validate(self.props).model_dump()
        return self


class Theme(Strict):
    primary: HexColor = "#2563eb"
    background: HexColor = "#ffffff"
    text: HexColor = "#111827"
    font: Literal["sans", "serif", "mono"] = "sans"


class Site(Strict):
    title: Short
    theme: Theme = Field(default_factory=Theme)
    sections: list[Section] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Site":
        ids = [s.id for s in self.sections]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate section ids: {dupes}")
        return self


def default_props(section_type: str, business: str = "your business") -> dict:
    """Sensible starter content for a section; used by add_section and the tests."""
    b = business
    return {
        "hero": {"headline": f"Welcome to {b}", "subheadline": f"Everything you need from {b}, in one place.",
                 "cta_label": "Get started", "cta_href": "#contact"},
        "text": {"heading": "About us", "body": f"{b.capitalize()} was built to make things simple, friendly and reliable."},
        "image": {"url": None, "alt": f"{b} photo", "caption": None},
        "features": {"heading": "Why choose us", "items": [
            {"title": "Expert team", "description": "Experienced people who care about the details."},
            {"title": "Flexible", "description": "Options that fit your schedule and budget."},
            {"title": "Trusted", "description": "Hundreds of happy customers and counting."}]},
        "testimonials": {"heading": "What our customers say", "items": [
            {"quote": "Exactly what I was looking for. Highly recommended.", "author": "Priya S.", "role": "Customer"},
            {"quote": "Friendly, professional and great value.", "author": "Daniel K.", "role": "Customer"}]},
        "pricing": {"heading": "Simple pricing", "plans": [
            {"name": "Starter", "price": "$19", "period": "month", "features": ["Core access", "Email support"], "highlighted": False},
            {"name": "Pro", "price": "$49", "period": "month", "features": ["Everything in Starter", "Priority support", "Extras"], "highlighted": True}]},
        "cta": {"heading": "Ready to get started?", "body": "Join us today.", "button_label": "Get started", "button_href": "#contact"},
        "contact": {"heading": "Contact us", "body": "Send a message and we will reply within a day.", "email": None,
                    "fields": ["name", "email", "message"], "submit_label": "Send"},
    }[section_type]


def _schema_prompt() -> str:
    """Compact schema description for prompts. Field names and optionality come from the models,
    so they cannot drift; the nested shapes and limits below are written by hand."""
    lines = ["Section types and their props (plain JSON, no extra keys):"]
    for name, model in PROPS_MODELS.items():
        fields = [f"{n}{'' if f.is_required() else '?'}" for n, f in model.model_fields.items()]
        lines.append(f"- {name}: {{{', '.join(fields)}}}")
    lines += [
        "features.items: [{title, description}] (1-8). testimonials.items: [{quote, author, role?}] (1-6). "
        "pricing.plans: [{name, price, period, features: [str], highlighted: bool}] (1-4). "
        "contact.fields: subset of ['name','email','phone','message']. image.url: https URL or null.",
        "Section: {id: unique kebab-case, type, props, style}.",
        "style: {background: '#rrggbb'|null, text_color: '#rrggbb'|null, padding: 'sm'|'md'|'lg', align: 'left'|'center'}.",
        "theme: {primary: '#rrggbb', background: '#rrggbb', text: '#rrggbb', font: 'sans'|'serif'|'mono'}.",
        "Colours MUST be hex strings, never names or rgb(). Short strings are at most 120 characters, long ones 800.",
    ]
    return "\n".join(lines)


SCHEMA_PROMPT = _schema_prompt()
