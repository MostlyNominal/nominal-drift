"""Serialization utilities for templates."""
import json
import yaml
from pathlib import Path
from nominal_drift.templates.base import BaseTemplate


def template_to_json(template: BaseTemplate) -> str:
    """Convert template to JSON string."""
    return template.model_dump_json(indent=2)


def template_from_json(json_str: str, template_class: type) -> BaseTemplate:
    """Load template from JSON string."""
    return template_class.model_validate_json(json_str)


def save_template(template: BaseTemplate, path: str) -> str:
    """Save template to .json or .yaml based on extension. Return absolute path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = template.model_dump()
    if p.suffix == ".yaml" or p.suffix == ".yml":
        p.write_text(yaml.dump(data, default_flow_style=False))
    else:
        p.write_text(json.dumps(data, indent=2))
    return str(p.resolve())


def load_template(path: str, template_class: type) -> BaseTemplate:
    """Load template from .json or .yaml."""
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(p.read_text())
    else:
        data = json.loads(p.read_text())
    return template_class.model_validate(data)
