"""Utility functions for parsing lightly annotated scripts."""
from typing import Dict, Any, List
import ast


def parse_script_line(line: str) -> Dict[str, Any]:
    """Parse a single script line formatted as 'Text => {key: value}'."""
    if "=>" not in line:
        return {"text": line.strip()}
    text, meta = line.split("=>", 1)
    text = text.strip()
    try:
        meta_dict = ast.literal_eval(meta.strip())
        if not isinstance(meta_dict, dict):
            meta_dict = {}
    except Exception:
        meta_dict = {}
    meta_dict["text"] = text
    return meta_dict


def parse_script(script: str) -> List[Dict[str, Any]]:
    """Parse multiple lines of annotated script into dictionaries."""
    return [parse_script_line(l) for l in script.strip().splitlines() if l.strip()]


if __name__ == "__main__":
    SAMPLE = (
        "INT. COFFEE SHOP - DAY => {'location': 'interior', 'place': 'coffee_shop', 'time': 'day'}\n"
        "John (scratching beard) => {'character': 'John', 'action': 'beard_scratch', 'emotion': 'pensive'}"
    )
    print(parse_script(SAMPLE))
