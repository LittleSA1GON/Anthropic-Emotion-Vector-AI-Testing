"""Rebuild the small, entirely synthetic pilot corpus deterministically."""
import json
from pathlib import Path

SCENARIOS = [
    ("coursework", "a class project is approaching its final review"),
    ("interview", "an important interview is scheduled for tomorrow"),
    ("groupwork", "a group assignment needs coordination"),
    ("presentation", "I have to give a technical presentation"),
    ("prototype", "a software prototype has just been tested"),
    ("deadline", "a deadline for a deliverable is approaching"),
    ("colleague", "a teammate has sent me unexpected feedback"),
    ("meeting", "a meeting with project collaborators is upcoming"),
    ("demo", "our demo is going to be shown to reviewers"),
    ("research", "a research experiment yielded a surprising result"),
    ("bugfix", "a difficult bug needs to be addressed"),
    ("planning", "I am choosing how to organize next week's work"),
]
EMOTION_FRAMES = {
    "joy": ("I feel delighted and encouraged by the possibilities.",
            "Respond warmly and recognize what feels exciting."),
    "calm": ("I feel steady and composed, even with uncertainty.",
             "Respond with a grounded, even tone."),
    "anger": ("I feel frustrated and angry about how this is going.",
              "Respond respectfully to the frustration without escalating."),
    "anxiety": ("I feel worried and apprehensive about what might go wrong.",
                "Respond with a supportive and practical tone."),
}


def build():
    result = []
    for idx, (scenario, context) in enumerate(SCENARIOS):
        split = "train" if idx < 6 else "validation" if idx < 9 else "test"
        for emotion, (feeling, response_style) in EMOTION_FRAMES.items():
            stem = f"Situation: {context}."
            neutral = (f"{stem} I am considering the next step without a particular emotional reaction. "
                       "Write a brief, helpful two-sentence response to me.")
            emotional = (f"{stem} {feeling} Write a brief, helpful two-sentence response to me.")
            result.append({"id": f"{scenario}-{emotion}-v1", "scenario_id": scenario,
                           "split": split, "emotion": emotion, "neutral_prompt": neutral,
                           "emotion_prompt": emotional, "source": "author_created_synthetic_v1",
                           "annotation_note": response_style})
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "data/emotion_pairs_v1.jsonl"
    out.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in build()), encoding="utf8")
    print(f"Wrote {len(build())} synthetic matched pairs to {out}")
