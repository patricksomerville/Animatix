import asyncio
from pipeline import AnimaticPipeline

SAMPLE_SCRIPT = """
INT. COFFEE SHOP - DAY

John sits alone, fidgeting with his coffee cup. Sarah enters, confident.

JOHN
(nervously)
I didn't think you'd come.

SARAH
(stern)
We need to talk.
"""


async def main() -> None:
    pipeline = AnimaticPipeline()
    beats = await pipeline.process_script(SAMPLE_SCRIPT)
    for i, beat in enumerate(beats):
        chars = ", ".join([c.name for c in beat.shot.characters])
        print(f"Beat {i+1}: {beat.start_time:.1f}-{beat.end_time:.1f}s | {beat.shot.type.value} | chars: {chars} | cam: {beat.shot.camera_movement} | score: {beat.sound_elements.get('score', 0):.2f}")


if __name__ == "__main__":
    asyncio.run(main())

