"""Small demo for Animatix components.

This script shows how to parse a tiny script sample and
build matching soundscapes for its dramatic moments.
"""
from script_reader import parse_script
from sound_design import build_soundscapes, Scene, Moment

SAMPLE = (
    "INT. COFFEE SHOP - DAY => {'location': 'interior', 'place': 'coffee_shop', 'time': 'day'}\n"
    "John (scratching beard) => {'character': 'John', 'action': 'beard_scratch', 'emotion': 'pensive'}\n"
)

if __name__ == "__main__":
    script = parse_script(SAMPLE)
    scene = Scene(key_moments=[Moment(start=10.0, turn_point=15.0)])
    sound = build_soundscapes(scene)
    print(script)
    print(sound)

