# HINTS_DETAILED (detailed tier)

Pick one tier per task: this file or `HINTS.md`, not both. Reading both wastes time. This tier shows the working core of each task with commentary explaining why each line is there. It withholds the function shell, the return assembly, and the glue, so completing the task still means reading and understanding the code, not pasting a finished function. The instructor solution notebook remains the only fully assembled artifact.

---

## Part A: run_evaluation

The working core is two calls:

```python
outputs = ls.generate_for_eval(model, tokenizer, eval_records, phase)
```

Why: this produces exactly one raw output string per eval record, in order. In offline mode it ignores the model and returns deterministic scripted text; in live mode it runs real greedy generation. Your code does not care which, and that indifference is the design: the evaluation logic is identical across backends.

```python
ls.evaluate_outputs(outputs, eval_records)
```

Why: scoring is a pure function of two aligned lists. It asserts the lengths match, so an off-by-one between generation and scoring fails loudly instead of silently mis-pairing records. The dict it returns is your function's return value.

You assemble: the def line, the docstring, and the return.

---

## Part B: audit_training_data

The working core, per record:

```python
label = rec["messages"][2]["content"]
ok, problems = ls.validate_output(label)
```

Why index 2: every record is system, user, assistant in that order, and the assistant message is the training label. The validator gives you a boolean and a list of human-readable problem strings; the strings are your classifier.

```python
if not ok:
    if any("not valid JSON" in p for p in problems):
        found["invalid_json"].append(rec["rid"])
    else:
        found["field_drift"].append(rec["rid"])
    continue
```

Why the string match: the validator reports a parse failure with a message that starts "not valid JSON". Anything else that made ok False on this dataset is a schema complaint, which here means the renamed key. The continue matters: a record lands in at most one bucket, and an unparseable label cannot be size-checked anyway.

```python
gold = json.loads(label)
stated = ls.extract_size_from_blurb(ls.blurb_of(rec))
labeled = gold["size_nominal_in"]
if (labeled is not None and stated is not None
        and abs(labeled - stated) > 0.01):
    found["unit_mismatch"].append(rec["rid"])
```

Why both None guards: unsized products carry null legitimately, and a blurb with no extractable size proves nothing. Only when both sides commit to a number and disagree beyond float tolerance do you have evidence of corruption. This is the defect the schema validator cannot see, which is the entire reason Part B exists.

You assemble: the def line, the docstring, the found dict initialization, the loop, and the sorted return.

---

## Part B: clean_training_data

The working core:

```python
drop = set(audit["invalid_json"])
drift = set(audit["field_drift"])
unit = set(audit["unit_mismatch"])
```

Why sets: membership tests run 450 times; O(1) lookups keep it clean.

```python
if rec["rid"] in drop:
    continue
rec = json.loads(json.dumps(rec))
```

Why the round-trip copy: the contract says do not mutate the input, and these records are pure JSON, so serialize-and-parse is a correct deep copy with zero imports. `copy.deepcopy` works too; this version also documents that the records are plain data.

```python
obj = json.loads(label)
obj = {("sku_mfg" if k == "mfg_sku" else k): v for k, v in obj.items()}
obj = {k: obj[k] for k in ls.REQUIRED_FIELDS}
rec["messages"][2]["content"] = json.dumps(obj)
```

Why the second comprehension: it rebuilds the dict in canonical field order after the rename, so every repaired label serializes with the same key order as every born-clean label. Consistent label formatting is a small gift to the fine-tune.

```python
obj["size_nominal_in"] = ls.extract_size_from_blurb(ls.blurb_of(rec))
```

Why trust the blurb: it is the primary source; the corrupted column was the export. The extractor was verified to read every sized blurb in this dataset correctly, which is why the repair is safe here and why you would demand the same verification before doing this to production data.

You assemble: the def line, the docstring, the branch structure connecting the pieces, the append, and the return.

---

## Part C: build_lora_config

The working core is the argument list:

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

Why each line: r is capacity, the rank of the A and B matrices. lora_alpha is influence; the forward pass scales the adapter path by alpha over r, so this pair gives scaling factor 2.0. target_modules is reach; Q and V attention projections are enough for format and style, which is this task. lora_dropout regularizes inside the adapter only. bias none keeps every base parameter frozen, including biases. task_type tells PEFT to build a causal language model wrapper; it is an enum member, and passing the string instead is the classic typo.

You assemble: the def line, the docstring, and the return.

---

## Part D: build_training_args

The working core:

```python
SFTConfig(
    output_dir="./lora_checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="no",
    eval_strategy="epoch",
    max_length=512,
    bf16=(ls.DEVICE == "mps"),
    use_cpu=(ls.DEVICE == "cpu"),
    report_to=[],
)
```

Why the notable lines: EPOCHS is 1 offline and 3 live, provided above the stub, so the same code serves both backends. Batch 4 with accumulation 4 gives effective batch 16 at quarter memory. Learning rate 2e-4 is roughly 10x a full fine-tune rate because far fewer parameters need a stronger per-step signal. eval_strategy is the current name; the old evaluation_strategy is gone from this trl and passing it fails. max_length is the current name; the old max_seq_length is likewise gone. bf16 needs MPS on macOS 14 or newer, hence the conditional. use_cpu satisfies current transformers on accelerator-free machines and is False on your Macs. report_to empty keeps experiment loggers out of a lab run.

You assemble: the def line, the docstring, and the return.

---

## Part D: run_training

The working core:

```python
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=splits["train"],
    eval_dataset=splits["test"],
    processing_class=tokenizer,
)
trainer.train()
```

Why processing_class: it is the current parameter name for the tokenizer across the Hugging Face trainer family; older tutorials show tokenizer= and are stale. Why the trainer accepts the messages column with no formatting code from you: SFTTrainer recognizes it natively, applies the tokenizer's chat template, and masks the loss so only assistant tokens are trained. Why return the trainer rather than the model: the trainer carries the log history Part D's checks read and the trained model Part E needs, both in one object.

You assemble: the def line, the docstring, and the return.

---

## Part E: evaluate_tuned

The working core is one call:

```python
run_evaluation(trainer.model, tokenizer, eval_records, "tuned_r16")
```

Why trainer.model: training mutated the adapter weights inside the PEFT model the trainer holds; that object is the trained artifact. Why your own Part A function: same records, same scoring, same code path before and after is what makes the comparison honest. The phase string labels the run and, in offline mode, selects the tuned script.

You assemble: the def line, the docstring, and the return.

---

## Part F: train_rank_variant (stretch)

The working core:

```python
fresh_base, fresh_tok = ls.load_base_model()
```

Why fresh: get_peft_model wraps in place. Wrapping the model you already trained would stack a second adapter onto the first, and the r=4 measurement would be contaminated by the r=16 weights. Fresh base, clean experiment.

```python
config = LoraConfig(
    r=r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
variant = get_peft_model(fresh_base, config)
```

Why only two values change: this is a controlled comparison. Same targets, same dropout, same data, same training args; the single manipulated variable is capacity (with alpha moved in lockstep to hold the scaling ratio at 2.0).

```python
variant_trainer = run_training(variant, build_training_args(), splits, fresh_tok)
```

Why reuse your own functions: that is the payoff of writing them as functions. The variant run is three of your existing building blocks and one config change.

You assemble: the def line, the docstring, the evaluation call with the given phase, and the return.

---

## Part G: capability_regression (stretch)

The working core:

```python
base_answers = ls.generate_capability(base_model, tokenizer, "baseline")
tuned_answers = ls.generate_capability(trained_model, tokenizer, "tuned_r16")
```

Why two models, two phases: the question is a delta, and a delta needs both endpoints measured the same way. The phase strings select the scripted answers in offline mode and are labels in live mode.

```python
ls.score_capability(base_answers)
ls.score_capability(tuned_answers)
```

Why a keyword score: it is deterministic, dependency-free, and sufficient to ring the alarm. It is deliberately crude; treating it as a real capability eval would be malpractice, and Week 6 builds the real thing.

You assemble: the def line, the docstring, and the return dict with keys baseline, tuned, and tuned_outputs (the raw tuned answers ride along so the driver can print the degraded one).
