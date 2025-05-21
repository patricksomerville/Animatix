"""Simple sound design utility functions."""
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Moment:
    start: float
    turn_point: float


@dataclass
class Scene:
    key_moments: List[Moment]


def build_soundscapes(scene: Scene) -> List[Dict[str, object]]:
    """Generate sound design events for key moments of a scene."""
    sound_map = []
    for moment in scene.key_moments:
        sound_map.append(
            {
                "start": moment.start - 3.5,
                "elements": {
                    "ambient_pressure": 0.7,
                    "score_tremolo": "c#_minor",
                    "foley_focus": "clock_ticks",
                },
            }
        )
        sound_map.append(
            {
                "start": moment.turn_point,
                "elements": {
                    "ambient_suckout": 1.0,
                    "score_sting": "violin_glissando_short",
                    "silence_length": 0.8,
                },
            }
        )
        sound_map.append(
            {
                "start": moment.turn_point + 0.8,
                "elements": {
                    "room_tone": "sudden_ear_ringing",
                    "score_undertow": "low_cello_drone",
                },
            }
        )
    return sound_map


if __name__ == "__main__":
    scene = Scene(key_moments=[Moment(start=10.0, turn_point=15.0)])
    print(build_soundscapes(scene))
