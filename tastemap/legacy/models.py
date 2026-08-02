from pydantic import BaseModel


class PreferenceClassification(BaseModel):
    primary_category: str
    context_categories: list[str] = []
    subcategories: list[str]
    tags: list[str]          # combined, stored in DB and shown in UI
    object_tags: list[str] = []
    sensory_tags: list[str] = []
    context_tags: list[str] = []
    social_tags: list[str] = []
    place_tags: list[str] = []
    preference_signal: str
    confidence: float
    classifier_source: str = "RULE_BASED_NO_API_KEY"


class PreferenceEntry(BaseModel):
    id: int | None
    raw_text: str
    rating: int
    primary_category: str
    context_categories: list[str] = []
    subcategories: list[str]
    tags: list[str]
    object_tags: list[str] = []
    sensory_tags: list[str] = []
    context_tags: list[str] = []
    social_tags: list[str] = []
    place_tags: list[str] = []
    preference_signal: str
    confidence: float
    created_at: str
    event_date: str | None = None
    event_date_precision: str = "none"
    time_perspective: str = "unspecified"
