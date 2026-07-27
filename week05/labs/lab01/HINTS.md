# HINTS (progressive tier)

Pick one tier per task: this file or `HINTS_DETAILED.md`, not both. Reading both wastes time. Use this tier when you want a nudge toward the approach; use the detailed tier when you want to study working code with commentary. Within this file, read one level at a time and stop as soon as you are unstuck.

---

## Part A: run_evaluation

**Level 1 (approach).** This is a pipeline of exactly two provided functions. One turns records into output strings, the other turns output strings into scores. Your function is the pipe between them.

**Level 2 (structure).**

```text
outputs = generate step, using all four of your arguments
return score step, using outputs and the eval records
```

**Level 3 (key line in context).** The generate step is:

```python
outputs = ls.generate_for_eval(model, tokenizer, eval_records, phase)
```

The score step takes `outputs` first and `eval_records` second, and its result is your return value.

---

## Part B: audit_training_data

**Level 1 (approach).** One pass over the records, three buckets, first matching bucket wins. The validator's `(ok, problems)` pair splits the first two buckets: not ok plus a JSON parse complaint is bucket one; not ok for any other reason is bucket two. Bucket three only applies to records the validator liked.

**Level 2 (structure).**

```text
for each record:
    label = the assistant message content
    ok, problems = validate it
    if not ok:
        if any problem string mentions the JSON parse failure -> invalid_json
        else -> field_drift
        continue
    parse the label, read its size
    read the blurb size with the provided extractor
    if both sizes exist and disagree beyond 0.01 -> unit_mismatch
sort each bucket before returning
```

**Level 3 (key lines in context).** The bucket split inside the not-ok branch:

```python
if any("not valid JSON" in p for p in problems):
    found["invalid_json"].append(rec["rid"])
else:
    found["field_drift"].append(rec["rid"])
```

And the mismatch test: `abs(labeled - stated) > 0.01`, guarded so that neither value is None.

---

## Part B: clean_training_data

**Level 1 (approach).** Three id sets from the audit, one pass over the records. Drop is a `continue`. Each repair parses the label, edits one thing, and re-serializes with `json.dumps`. Copy before you edit so the input list survives.

**Level 2 (structure).**

```text
build three sets from the audit dict
for each record:
    if rid in the drop set: skip it
    deep copy the record
    if rid in the drift set: parse, rename the bad key, rebuild the label
    else if rid in the unit set: parse, overwrite the size from the blurb, rebuild
    append the (possibly repaired) copy
```

**Level 3 (key lines in context).** The rename, done as a rebuild so the bad key disappears:

```python
obj = {("sku_mfg" if k == "mfg_sku" else k): v for k, v in obj.items()}
```

A cheap deep copy that works on pure-JSON records: `json.loads(json.dumps(rec))`.

---

## Part C: build_lora_config

**Level 1 (approach).** Every value you need is in the table directly above the stub. This task is about knowing what each argument means, not about discovering values. Say the five concepts out loud as you type: capacity, influence, reach, regularization, wrapper type.

**Level 2 (structure).** One `LoraConfig(...)` call, six keyword arguments, returned directly. `target_modules` takes a list of module name strings. `task_type` takes an enum member, not a string.

**Level 3 (key line in context).** The one argument people mistype:

```python
task_type=TaskType.CAUSAL_LM,
```

---

## Part D: build_training_args

**Level 1 (approach).** Transcribe the contract list into one `SFTConfig(...)` call. Two of the values are expressions, not literals: the bf16 flag and the use_cpu flag both depend on `ls.DEVICE`.

**Level 2 (structure).** Eleven keyword arguments in the order listed. The two conditionals:

```text
bf16 = (device is mps)
use_cpu = (device is cpu)
```

**Level 3 (key lines in context).**

```python
bf16=(ls.DEVICE == "mps"),
use_cpu=(ls.DEVICE == "cpu"),
```

---

## Part D: run_training

**Level 1 (approach).** Construct the trainer, call one method on it, return the trainer itself. Five constructor arguments: the model, the args, the two dataset splits, and the tokenizer under its modern parameter name.

**Level 2 (structure).**

```text
trainer = SFTTrainer(model, args, train split, eval split, tokenizer as processing_class)
train it
return it
```

**Level 3 (key line in context).** The parameter name that changed across trl versions:

```python
processing_class=tokenizer,
```

The splits ride in as `train_dataset=splits["train"]` and `eval_dataset=splits["test"]`.

---

## Part E: evaluate_tuned

**Level 1 (approach).** You already wrote the evaluation in Part A. The only new fact is where the trained model lives after training finishes.

**Level 2 (structure).** One line: call your Part A function with the trained model, the tokenizer, the eval records, and the tuned phase label.

**Level 3 (key fact).** The trained PEFT model is `trainer.model`. The phase string is `"tuned_r16"`.

---

## Part F: train_rank_variant (stretch)

**Level 1 (approach).** This is Parts C, D, and E replayed with two numbers changed. The one new rule: load a fresh base model first. Wrapping an already-wrapped model stacks adapters and corrupts the comparison.

**Level 2 (structure).**

```text
fresh base and tokenizer from ls.load_base_model()
LoraConfig exactly like Part C but with the r and lora_alpha arguments
wrap the fresh base
train with your existing run_training and build_training_args
evaluate with your existing run_evaluation and the given phase
```

**Level 3 (key line in context).** The fresh start:

```python
fresh_base, fresh_tok = ls.load_base_model()
```

Everything after it is your own Part C, D, and A code with `r=r, lora_alpha=lora_alpha` threaded through.

---

## Part G: capability_regression (stretch)

**Level 1 (approach).** Two generate calls with different models and phase labels, two score calls, one dict assembled from the results.

**Level 2 (structure).**

```text
base answers   = generate capability with the base model, phase "baseline"
tuned answers  = generate capability with the trained model, phase "tuned_r16"
return the two scores plus the tuned answers, under the three required keys
```

**Level 3 (key line in context).**

```python
base_answers = ls.generate_capability(base_model, tokenizer, "baseline")
```

The tuned call mirrors it with the trained model and the tuned phase. Score each list with `ls.score_capability`.
