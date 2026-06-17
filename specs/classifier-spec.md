# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
Request two keyed lines, in this exact order:

Label: <one of: interview, solo, panel, narrative>
Reasoning: <one or two sentences>

Tradeoffs considered:
- Bare label on its own line: trivial to parse, but discards the reasoning
  the return dict requires.
- JSON: structurally clean, but llama-3.3-70b often wraps it in ```json
  fences or adds a preamble ("Here is the classification:"). A single stray
  character breaks json.loads, forcing fence-stripping plus try/except.
- Keyed lines (chosen): tolerant of surrounding prose and whitespace. Parse
  by scanning lines case-insensitively for the one starting with "label:"
  and "reasoning:", then split on the first ":". Gives both fields cleanly.
```

---

**Edge cases to handle in the prompt:**

```
- Empty labeled_examples: degrade to zero-shot. Still emit the task
  instructions and the four label definitions, but skip the examples
  section entirely rather than emitting an empty/garbled "Examples:" block
  that would confuse the model.
- Very short or thin description: still send it. The label definitions plus
  the keyed output format push the model to commit to a best-guess label
  instead of refusing; if it returns junk, classify_episode() validation
  maps it to "unknown".
- Description that itself contains "Label:" or the "---" delimiter: wrap the
  to-classify description in a clearly demarcated block (e.g. its own
  "Episode to classify:" header) so it is not mistaken for an example or a
  separator.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
The prompt requests two keyed lines: "Label: X" and "Reasoning: Y".

Parse by:
1. Split the response text into lines.
2. For each line, strip() it and lowercase a copy to test the prefix.
3. The line starting with "label:" -> take everything after the first ":",
   strip(), lowercase() -> candidate label.
4. The line starting with "reasoning:" -> take everything after the first
   ":", strip() (keep original case) -> reasoning.
5. If no "reasoning:" line is found, fall back to the full response text so
   we never lose the model's explanation.
```

---

**Step 4 — Validate the label:**

```
Compare the parsed candidate label (already lowercased and stripped) against
VALID_LABELS. If it is one of them, use it. Otherwise set label to "unknown".
This catches the model inventing a label, adding punctuation, or returning a
malformed/empty line. The reasoning is kept regardless so we can inspect why.
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the LLM call and parsing in a try/except. Things that can go wrong:
- Network/API error, timeout, rate limit, bad API key (raises from the client).
- Empty or None response content.
- A response with no parseable "Label:" line (handled by the "unknown"
  fallback, not an exception).

On any exception, return a well-formed dict so the eval loop never crashes:
  {"label": "unknown", "reasoning": f"Error: {e}"}
One bad response degrades to a single "unknown" row instead of aborting all
20 calls.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: [title]
Raw response text: [paste it here]
```

**How did you parse the label out of the response?**

```
[describe the string operations — strip, split, lower, etc.]
```

**Did any episodes return `"unknown"`? If so, why?**

```
[yes / no — if yes, what did the raw response look like?]
```

**One thing about the output format that surprised you:**

```
[your answer here]
```
