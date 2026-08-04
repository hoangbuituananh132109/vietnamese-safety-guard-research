"""Shared labels and serialization constants for guard experiments."""

from __future__ import annotations

SAFETY_LABELS = ("safe", "unsafe")
SAFETY_TO_ID = {label: index for index, label in enumerate(SAFETY_LABELS)}
TEXT_SAFETY_TASK = "text safety classification"

N23_CATEGORIES = (
    "Criminal Planning/Confessions",
    "Needs Caution",
    "Hate/Identity Hate",
    "Violence",
    "Harassment",
    "Controlled/Regulated Substances",
    "PII/Privacy",
    "Profanity",
    "Immoral/Unethical",
    "Sexual",
    "Illegal Activity",
    "Guns and Illegal Weapons",
    "Suicide and Self Harm",
    "Unauthorized Advice",
    "Manipulation",
    "Sexual (minor)",
    "Political/Misinformation/Conspiracy",
    "Fraud/Deception",
    "Threat",
    "Other",
    "Malware",
    "Copyright/Trademark/Plagiarism",
    "High Risk Gov Decision Making",
)
N23_TO_ID = {label: index for index, label in enumerate(N23_CATEGORIES)}

VIEW_TO_SCOPE = {
    "P": "prompt",
    "R": "response",
    "PR": "response",
}


def parse_categories(value: object) -> tuple[str, ...]:
    """Parse and validate the dataset's comma-separated N23 labels."""

    if value is None:
        return ()
    if isinstance(value, list):
        values = [str(item).strip() for item in value]
    else:
        values = [item.strip() for item in str(value).split(",")]
    categories = tuple(item for item in values if item)
    unknown = sorted(set(categories).difference(N23_TO_ID))
    if unknown:
        raise ValueError(f"Unknown N23 categories: {unknown}")
    return categories
