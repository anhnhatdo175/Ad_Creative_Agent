from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class ProductInput(BaseModel):
    url: HttpUrl | None = None
    description: str | None = Field(default=None, max_length=10000)
    idempotency_key: str | None = Field(default=None, max_length=128)


class ProductFact(BaseModel):
    fact_id: str
    label: str
    value: str
    source_snippet: str


class ProductFacts(BaseModel):
    name: str
    category: str
    description: str
    features: list[str] = Field(default_factory=list)
    price: str | None = None
    rating: str | None = None
    reviews: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    source_url: str | None = None
    facts: list[ProductFact]


class StoryboardScene(BaseModel):
    visual: str
    voiceover: str
    duration: int


class Variant(BaseModel):
    id: str
    angle: Literal["pain_point", "social_proof", "curiosity", "benefit", "urgency", "comparison"]
    caption: str
    script: str
    storyboard: list[StoryboardScene]
    image_prompts: list[str]
    fact_ids: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class GeneratedVariant(BaseModel):
    angle: Literal["pain_point", "social_proof", "curiosity", "benefit", "urgency", "comparison"]
    caption: str
    script: str
    storyboard: list[StoryboardScene] = Field(min_length=3, max_length=4)
    image_prompts: list[str] = Field(min_length=1, max_length=3)
    fact_ids: list[str] = Field(min_length=1)


class GeneratedAdPack(BaseModel):
    variants: list[GeneratedVariant] = Field(min_length=3, max_length=3)


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "partial", "failed"]
    steps: list[dict]
    variants: list[Variant] = []
    error: str | None = None
