import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    task_instruction = (
        "You are classifying podcast episodes by their format. Classify the "
        "episode into exactly one of these four labels:\n"
        "\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — "
        "no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often "
        "debating or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, "
        "archival audio, reporting — with a clear narrative arc\n"
        "\n"
        "Return only the label and your reasoning. "
        "Do not explain the taxonomy."
    )

    output_format = (
        "Respond with exactly two lines, in this order:\n"
        "Label: <one of: interview, solo, panel, narrative>\n"
        "Reasoning: <one or two sentences>"
    )

    parts = [task_instruction]

    # Few-shot examples. If there are none, degrade to zero-shot by skipping
    # the examples section entirely rather than emitting an empty block.
    if labeled_examples:
        parts.append("Here are labeled examples:")
        example_blocks = []
        for ex in labeled_examples:
            example_blocks.append(
                f"Title: {ex['title']}\n"
                f"Description: {ex['description']}\n"
                f"Label: {ex['label']}"
            )
        parts.append("\n\n---\n\n".join(example_blocks))

    # Demarcate the episode to classify so its text can't be mistaken for an
    # example or a delimiter, even if it contains "Label:" or "---".
    parts.append(
        "Episode to classify:\n"
        f"Description: {description}"
    )

    parts.append(
        "Classify the episode above. Return your answer in the format below:\n"
        f"{output_format}"
    )

    return "\n\n".join(parts)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    try:
        # Step 1 — build the prompt.
        prompt = build_few_shot_prompt(labeled_examples, description)

        # Step 2 — send it to the LLM.
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        text = response.choices[0].message.content or ""

        # Step 3 — parse the keyed-line response.
        candidate_label = ""
        reasoning = ""
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if lowered.startswith("label:"):
                candidate_label = stripped.split(":", 1)[1].strip().lower()
            elif lowered.startswith("reasoning:"):
                reasoning = stripped.split(":", 1)[1].strip()

        # Fall back to the full response so we never lose the explanation.
        if not reasoning:
            reasoning = text.strip()

        # Step 4 — validate the label.
        label = (
            candidate_label if candidate_label in VALID_LABELS else "unknown"
        )

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        # Step 5 — one bad call degrades to a single "unknown" row instead of
        # crashing the whole evaluation loop.
        return {"label": "unknown", "reasoning": f"Error: {e}"}
