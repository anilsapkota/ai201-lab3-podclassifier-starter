# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
Accuracy = (number of predictions that exactly match the ground-truth label
at the same position) / (total number of predictions).

"Correct" means predictions[i] == ground_truth[i] — an exact string match,
case-sensitive, including the "unknown" label (an "unknown" prediction is
counted as wrong unless the ground truth is literally "unknown", which it
never is). The denominator is the total number of episodes evaluated, i.e.
len(predictions).
```

---

**Step-by-step logic:**

```
1. If predictions is empty, return 0.0 (handle the edge case first).
2. Pair up predictions and ground_truth position-by-position (zip).
3. Count the pairs where the two labels are equal.
4. Divide that count by len(predictions) and return as a float.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. With no episodes there is nothing to be right about, and
dividing by len(predictions) == 0 would raise ZeroDivisionError. Returning
0.0 keeps the return type a float and prevents a crash. (We assume the two
lists are always the same length, as guaranteed by run_evaluation().)
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

index 0: interview == interview  -> correct
index 1: solo      == solo        -> correct
index 2: panel     != solo        -> wrong
index 3: interview != narrative   -> wrong

correct = 2, total = 4
compute_accuracy() returns 2 / 4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
For the "interview" class, an episode counts as correct when its ground-truth
label is "interview" AND the prediction for that same episode is also
"interview". The class is defined by the GROUND TRUTH, not the prediction:
we only look at episodes whose true label is "interview", then ask whether
each was predicted correctly. A "solo" episode wrongly predicted as
"interview" does NOT count toward the interview class (it counts against the
solo class as a miss).
```

---

**What does "total" mean for a given class?**

```
"total" for a class is the number of episodes whose GROUND-TRUTH label is
that class — not the total number of predictions, and not how many times the
model predicted that class. For VALID_LABELS the per-class totals sum to the
overall number of episodes (assuming every ground-truth label is valid).
```

---

**Step-by-step logic:**

```
1. Initialize a stats dict, one entry per label in VALID_LABELS, each
   {"correct": 0, "total": 0, "accuracy": 0.0}.
2. Loop over the (predicted, truth) pairs with zip(predictions, ground_truth).
3. For each pair:
   - Skip if truth is not in VALID_LABELS (defensive; shouldn't happen).
   - Increment stats[truth]["total"] by 1.
   - If predicted == truth, increment stats[truth]["correct"] by 1.
4. After the loop, for each label compute accuracy = correct / total, but
   set it to 0.0 when total == 0 (avoid division by zero).
5. Return the stats dict.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
accuracy = 0.0 (as the evaluate.py docstring specifies: "0.0 if total is 0").
There are no episodes of that class to score, so any other value would be
misleading and correct / total would divide by zero. correct and total both
stay 0, and accuracy is reported as 0.0.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Group episodes by their ground-truth label, then check each prediction:

- interview: idx 0 truth=interview, pred=interview -> 1/1
- solo:      idx 1 truth=solo, pred=interview (wrong);
             idx 2 truth=solo, pred=solo (right)        -> 1/2
- panel:     idx 3 truth=panel, pred=panel              -> 1/1
- narrative: idx 4 truth=narrative, pred=panel (wrong)  -> 0/1

label       correct  total  accuracy
----------  -------  -----  --------
interview      1       1      1.0
solo           1       2      0.5
panel          1       1      1.0
narrative      0       1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
