"""JSON schemas for every structured LLM output, plus a small local validator.

Provider-side structured outputs constrain the format; we still validate locally
because (a) not every provider/model enforces every keyword and (b) we add
semantic checks the schema language cannot express.
"""
from __future__ import annotations

from typing import Any


class ValidationError(ValueError):
    pass


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": required if required is not None else list(props),
        "additionalProperties": False,
    }


DECISION = _obj({
    "current_belief": {"type": "string",
                       "description": "What the recorded results currently support, citing experiment ids (#n)"},
    "evidence_used": {"type": "array", "items": {"type": "integer"},
                      "description": "Experiment ids this round's reasoning rests on"},
    "alternatives_considered": {
        "type": "array",
        "items": _obj({"idea": {"type": "string"},
                       "why_not_now": {"type": "string", "description": "Why it is not proposed this round"}}),
        "description": "Directions considered but NOT proposed this round (at most 4)"},
    "assumptions": {"type": "array", "items": {"type": "string"},
                    "description": "Assumptions this round relies on or now questions (at most 3)"},
    "uncertain": {"type": "array", "items": {"type": "string"},
                  "description": "What remains uncertain after the results so far (at most 3)"},
    "why_these": {"type": "string", "description": "Why the proposed hypotheses are the most useful next tests"},
})

HYPOTHESES = _obj({
    "decision": DECISION,
    "hypotheses": {
        "type": "array",
        "items": _obj({
            "title": {"type": "string", "description": "Short unique name, <= 80 chars"},
            "rationale": {"type": "string", "description": "Why this should beat the current best, citing prior experiment ids"},
            "method": {"type": "string", "description": "Concrete, implementable description of the experiment"},
            "expected_metric": {"type": "number", "description": "Predicted primary metric on the validation split"},
            "priority": {"type": "number", "description": "0..1, higher runs first"},
            "parent_experiment_ids": {"type": "array", "items": {"type": "integer"},
                                      "description": "Experiment ids whose results motivated this hypothesis"},
            "falsified_if": {"type": "string",
                             "description": "The measured result that would count against this hypothesis"},
        }),
    },
})

CODE = _obj({
    "code": {"type": "string", "description": "Complete single-file Python program"},
    "explanation": {"type": "string"},
    "requested_permissions": {
        "type": "array", "items": {"type": "string"},
        "description": "Extra capabilities wanted (e.g. network, packages). Never granted automatically.",
    },
})

ANALYSIS = _obj({
    "verdict": {"type": "string", "enum": ["supported", "refuted", "inconclusive"]},
    "analysis": {"type": "string"},
    "lessons": {"type": "array", "items": {"type": "string"}},
})


_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "integer": int, "number": (int, float),
}


def validate(data: Any, schema: dict, path: str = "$") -> None:
    """Validate the subset of JSON Schema used above. Raises ValidationError."""
    t = schema.get("type")
    if t:
        py = _TYPES[t]
        # bool is a subclass of int; reject it for numeric types.
        if not isinstance(data, py) or (t in ("integer", "number") and isinstance(data, bool)):
            raise ValidationError(f"{path}: expected {t}, got {type(data).__name__}")
    if "enum" in schema and data not in schema["enum"]:
        raise ValidationError(f"{path}: {data!r} not in {schema['enum']}")
    if t == "object":
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in data:
                raise ValidationError(f"{path}: missing required key {key!r}")
        if schema.get("additionalProperties") is False:
            extra = set(data) - set(props)
            if extra:
                raise ValidationError(f"{path}: unexpected keys {sorted(extra)}")
        for key, sub in props.items():
            if key in data:
                validate(data[key], sub, f"{path}.{key}")
    elif t == "array":
        for i, item in enumerate(data):
            validate(item, schema.get("items", {}), f"{path}[{i}]")
