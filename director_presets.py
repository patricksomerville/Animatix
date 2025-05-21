"""Collection of example directing style presets."""
from typing import Callable, Dict, List


def _release_conflict(turn: bool) -> str:
    return "release TO conflict" if turn else "constrict"


def _tarantino_moment(_: bool) -> str:
    return "comic_beat_post_violence"


DIRECTING_DNA: Dict[str, Dict[str, object]] = {
    "spikeLee": {
        "coverage": [
            {"shot": "doubles_2s_dutch", "freq": 0.4},
            {"shot": "dolly_push_tight", "freq": 0.3},
        ],
        "sound": {
            "scoreCut": "syncopated_jazz_break",
            "dialogueMix": {"reverbDecay": 2.1},
        },
        "momentEmphasis": _release_conflict,
    },
    "tarantino": {
        "coverage": [
            {"shot": "trunk_shot", "freq": 0.1},
            {"shot": "two_shot_low", "freq": 0.6},
        ],
        "sound": {
            "scoreCut": "surf_rock_envelope",
            "dialogueMix": {"reverbDecay": 0.3},
        },
        "momentEmphasis": _tarantino_moment,
    },
}

if __name__ == "__main__":
    for name, preset in DIRECTING_DNA.items():
        print(name, "->", preset["coverage"])
