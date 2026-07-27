# Week 4 Knowledge Check: Solution Key

**AI Engineering Academy | Gamut Technology Services | Instructor-facing. Do not distribute to students.**

Answer distribution: A appears 1 time, B 4 times, C 4 times, D 1 time. No positional pattern, and the correct answer is not consistently the longest option.

Question types: code questions are 2, 3, 5, 6, and 7. Concept questions are 1, 4, 8, 9, and 10.

| Q | Answer | Type | Maps to |
|---|--------|------|---------|
| 1 | B | concept | Module 01, causal masking in decoder-only models |
| 2 | A | code | Module 01, logits, softmax, temperature |
| 3 | B | code | Module 02, logit attribution, cache versus model ownership |
| 4 | C | concept | Module 01, ablation and the Hydra effect |
| 5 | B | code | Module 02, hook calling convention |
| 6 | B | code | Module 02, activation cache tuple keys |
| 7 | B | code | Module 02, gradient-based saliency, leaf tensors |
| 8 | C | concept | Modules 01 and 02, correlational versus causal evidence |
| 9 | B | concept | Module 00, grouped-query attention |
| 10 | C | concept | Modules 00 and 02, TransformerBridge compatibility mode |

---

### Question 1 - Answer: B

Decoder-only models apply a causal mask so position `i` can attend to positions `0` through `i` only. Attention to future positions is set to negative infinity before the softmax, which drives those weights to zero. This is what makes autoregressive generation coherent: the representation at each position cannot depend on tokens the model has not emitted yet.

Why the distractors are wrong. A is the specific misconception this question targets and directly matches the Module 01 slide 5 wording we flagged. The mask is architectural and is applied at inference exactly as in training. Encoder models such as BERT are bidirectional, which is where the "every token attends to every other token" phrasing comes from, so a student who learned attention through BERT will find A plausible. C invents a per-head isolation rule; heads operate over the full masked sequence and are concatenated. D describes something closer to an RNN and misstates why position embeddings exist. Position embeddings are needed precisely because attention is otherwise order-agnostic across all visible positions.

---

### Question 2 - Answer: A (code)

Temperature divides the logits before the softmax. Division by a positive constant is monotonic, so it never reorders the logits. Index 0 has the largest logit at every temperature, which is why `argmax=0` prints three times. Lower temperature sharpens the distribution toward the top token and higher temperature flattens it, but the ranking is fixed. The practical consequence for engineers: if you decode greedily, temperature is inert. It only changes behavior when you actually sample.

Verified: the printed output is the real result of running this code, and argmax stays at index 0 across temperatures from 0.1 through 10.0.

Why the distractors are wrong. B and C both claim temperature reorders tokens, which is the central misconception. B is the folk explanation of why high temperature is "creative," and it is wrong about the mechanism. Higher temperature makes lower-ranked tokens more likely to be *sampled*, but it never makes them the argmax. C states the reordering claim plainly and is contradicted by the output printed directly above it, which is the reading-carefully test. D reverses the order of operations. Temperature is applied to logits before the softmax, not to probabilities after it.

Live follow-up worth 60 seconds: ask what temperature 0 means mechanically. Division by zero, so implementations special-case it to argmax rather than computing it.

---

### Question 3 - Answer: B (code)

`accumulated_resid` is a method on `ActivationCache`, the second value returned by `run_with_cache`. It does not exist on the model or on the bridge, so calling it on `model` raises `AttributeError`. The method also takes no `tokens` argument and no `return_type` argument, and it returns residual stream contributions rather than logits. Logit attribution is a different method, `cache.logit_attrs`, and the per-component split is `cache.decompose_resid`.

The correct shape of this workflow:

```python
tokens = model.to_tokens("The cat sat on the")
logits, cache = model.run_with_cache(tokens)
target = model.to_single_token(" cat")
per_component, labels = cache.decompose_resid(return_labels=True)
attribution = cache.logit_attrs(per_component, tokens=target)
```

Verified against TransformerLens 3.5.1: `accumulated_resid`, `decompose_resid`, `logit_attrs`, and `apply_ln_to_stack` are all present on `ActivationCache` and all absent from `TransformerBridge`. The real `accumulated_resid` signature accepts `layer`, `incl_mid`, `apply_ln`, `pos_slice`, `mlp_input`, and `return_labels`, with no `tokens` and no `return_type`.

Why the distractors are wrong. A is the assumption the student made. C invents a plausible-sounding default-layer behavior and is attractive because `layer` genuinely is an optional parameter on the real method, just on the wrong object. D correctly senses that an argument is wrong but names the wrong failure and the wrong error type. The call never gets far enough to validate arguments, because attribute lookup fails first. This ordering point is worth making explicitly: `AttributeError` beats `TypeError` because Python resolves the attribute before binding arguments.

---

### Question 4 - Answer: C

The Hydra effect is the observed tendency of a network to compensate when a component is removed. Other heads and MLP layers partially take over the ablated head's function, so the measured loss increase is a lower bound on the component's contribution in the intact model. A small delta is therefore weak evidence of unimportance. Reach for a stronger design before concluding: compare against mean or resample ablation rather than zero ablation, measure KL against the clean distribution rather than task loss alone, and use activation patching to localize the effect positionally.

Why the distractors are wrong. A overgeneralizes into a claim that zero ablation is uninformative, which is too strong. It is informative but biased downward. B is the most tempting distractor because switching to KL genuinely is good practice and appears in the Module 02 ablation code. The flaw is the word "only." KL is a better-behaved metric than raw loss, but it does not remove the compensation effect, so it does not rescue the teammate's inference. D inverts the epistemics. Ablation is the intervention, so it is the causal tool. Attention weights are the correlational signal, which is the subject of Question 8.

---

### Question 5 - Answer: B (code)

TransformerLens invokes forward hooks as `hook(activation, hook=hook_point)`. The hook function must therefore accept the activation positionally and a `hook` keyword. A one-argument lambda fails on the keyword. The exact error is `TypeError: <lambda>() got an unexpected keyword argument 'hook'`.

The corrected form, which also fixes a second latent problem:

```python
from functools import partial

def zero_head(activation, hook, head_index):
    activation[:, :, head_index, :] = 0.0
    return activation

model.run_with_hooks(
    tokens,
    fwd_hooks=[("blocks.3.attn.hook_z", partial(zero_head, head_index=7))],
)
```

Verified against TransformerLens 3.5.1 by inspecting `HookPoint.add_hook`, which calls `hook(module_output, hook=self)`, and by attaching both forms to a live `HookPoint`. The one-argument lambda raised the `TypeError` quoted above and the two-argument function ran and zeroed the tensor.

Worth naming in the debrief: even after fixing the signature, the original body is questionable. `zero_()` is an in-place op on a slice and returns the slice, not the full activation, so a hook written that way returns the wrong tensor shape when the return value is used. `partial` is the standard way to bind the head index, and it is the pattern used in the Module 02 ablation slides.

Why the distractors are wrong. A is what the student expects. C is plausible to anyone who has not confirmed hook names, but `blocks.3.attn.hook_z` is a valid alias once compatibility mode is enabled. D invents a broadcasting behavior that does not exist.

---

### Question 6 - Answer: B (code)

This is the silent-failure question, and it is the one most worth spending debrief time on. The tuple key form resolves through `utils.get_act_name(name, layer, layer_type)`. The third positional slot is `layer_type`, not a head index. Passing `3` there is simply ignored, so `cache["pattern", 0, 3]` resolves to the same hook name as `cache["pattern", 0]` and returns the pattern for every head in layer 0. No error, no warning, and a shape of `[batch, head, query_pos, key_pos]` rather than the per-head tensor the student thinks they have. Every downstream number computed from it is wrong.

Head selection is a tensor index, not a key argument:

```python
head_3_pattern = cache["pattern", 0][:, 3]   # [batch, query_pos, key_pos]
```

Verified against TransformerLens 3.5.1: `utils.get_act_name("pattern", 0, 3)` returns `blocks.0.attn.hook_pattern`, identical to `utils.get_act_name("pattern", 0)`.

Why the distractors are wrong. A is the belief being tested. C invents a transposition, which is attractive because the numbers 0 and 3 are both present and swapping them feels like the kind of bug that would be here. D invents an indexing rule. The teaching point is that the printed shape is the tell, which is why the question shows a `print` of `.shape` without giving the value away. Students who mentally run the shape catch it.

---

### Question 7 - Answer: B (code)

`actual` carries grad history, so `actual - baseline` and the scaled sum are non-leaf tensors with a `grad_fn`. PyTorch refuses to change the `requires_grad` flag on a non-leaf and raises `RuntimeError: you can only change requires_grad flags of leaf variables.` The fix is to break the graph and create a fresh leaf on each step:

```python
interp = (baseline + alpha * (actual - baseline)).detach()
interp.requires_grad_(True)
```

Then `.grad` populates after `backward()`. Also call `model.zero_grad(set_to_none=True)` inside the loop so gradients do not accumulate across steps.

Verified with torch 2.13.0. With `actual` produced by a real `nn.Embedding` forward pass, `actual.is_leaf` is `False` with `grad_fn` `EmbeddingBackward0`, `interpolated.is_leaf` is `False`, and the assignment raises the `RuntimeError` quoted above. After `detach()` plus `requires_grad_(True)`, `is_leaf` is `True` and `.grad` is populated with the right shape.

One nuance for a sharp student who tests this in isolation and reports it working: if `actual` is detached or was built under `torch.no_grad()`, the intermediate is already a leaf and the assignment succeeds. The failure depends on the tensor's provenance, which is exactly why this bug survives casual testing and appears in real interpretability code.

Why the distractors are wrong. C is the strongest distractor because the silent `.grad is None` outcome is a real PyTorch behavior for non-leaf tensors, just the one that occurs when you *omit* the assignment rather than make it. Students who know that gotcha will reach for C, so distinguishing the loud failure from the quiet one is the discrimination this question is built on. A misstates `zeros_like`, which does not propagate `requires_grad`. D is arithmetically true that the first tensor is all zeros, but an all-zero tensor carries gradients perfectly well.

Separate currency note for the instructor: this snippet also uses `start_at_layer=0`, which is not implemented on the bridge. The Module 02 corrections replace it with `hook_embed` injection. That is out of scope for the question as asked, but expect it to come up, and it is a fair bonus point if a student names it.

---

### Question 8 - Answer: C

Attention weight tells you where a head looked, not what the output depended on. A head can attend strongly to a token whose value vector contributes little, the information can be overwritten downstream, or another path can carry the actual effect. Converting a "looked at" observation into a "mattered" claim requires an intervention: ablate the component and measure the change, or patch a clean activation into a corrupted run and see whether the behavior flips. This correlational-versus-causal distinction is the spine of Week 4 and the reason the module pairs attention visualization with ablation rather than teaching visualization alone.

Why the distractors are wrong. A is the intuitive reading and is the claim the week is built to dismantle. B invents a threshold, which is the most seductive option for engineers who want a decision rule. There is no such threshold, and the number 0.87 in the stem is bait for exactly that instinct. D overcorrects. Attention weights are not meaningless. They are a real signal and a legitimate starting hypothesis, just not proof.

Engineer framing for the debrief: attention is a profiler sample showing which function was on the stack. Ablation is deleting the function and seeing whether the program still works.

---

### Question 9 - Answer: B

Grouped-query attention keeps the full set of query heads and shares each key-value projection across a group of them. With 32 query heads and 8 key-value heads, four query heads share each key-value pair, so the key-value cache shrinks by a factor of four. That cache is the dominant memory cost at long context during inference, which is what the design targets. The tradeoff is reduced diversity in the key-value projections.

Why the distractors are wrong. A is the most common wrong answer and the reason this question is here. It reads the smaller number as the head count, concluding the model runs 8 heads. It still runs 32 query heads, and the attention compute is largely unchanged. The saving is memory, not compute. C invents sequence-length truncation and describes something closer to a sliding window. D describes tensor parallelism, a real technique unrelated to the query-to-key-value head ratio.

---

### Question 10 - Answer: C

`TransformerBridge` wraps the native HuggingFace model and preserves its raw weights and naming by default. `enable_compatibility_mode()` folds LayerNorm into adjacent weights, centers the weights, and registers the legacy hook aliases the interpretability recipes reference. Without it, hook names such as `blocks.3.attn.hook_z` may not resolve, and the logit lens, direct logit attribution, and residual-stream decomposition produce numbers that do not match what those recipes assume, because they were derived under folded-LayerNorm numerics. It is a one-shot call that mutates weights in place, so call it immediately after boot and only once.

Why the distractors are wrong. A overstates it. Boot succeeds on its own and plain generation works fine, which is precisely why the omission is easy to miss. B is the reassuring answer and is wrong in the way that matters most here: the results differ numerically, not just in speed. That is the difference between a lab that produces defensible attribution numbers and one that quietly produces different ones. D invents an architecture-specific carve-out.

---

## Scoring and use

Suggested cut line for solid understanding is 7 of 10. This quiz is harder than the Week 3 check, which matches the material.

The three questions most likely to separate the room are 6 (silently wrong, no error), 7 (loud failure versus quiet failure), and 9 (the grouped-query head-count trap). Question 6 is the most valuable of the three, because it is the only item where the code runs cleanly and still produces a wrong answer, which is the habit Week 4 is meant to build.

Fast debrief order if time is short: 6, 8, 4, 2, then the rest. Those four carry the durable lessons, which are that clean execution is not correctness, that attention is not an explanation, that compensation biases ablation results downward, and that temperature does not reorder tokens.

---

## Verification ledger

Every code-bearing claim was executed rather than recalled.

| Claim | How verified | Result |
|---|---|---|
| Softmax values and argmax invariance (Q2) | Ran the printed snippet; swept temperature 0.1 to 10.0 | Output as printed; argmax stayed at index 0 at every temperature |
| `accumulated_resid` ownership and signature (Q3) | Attribute checks and `inspect.signature` on TransformerLens 3.5.1 | Present on `ActivationCache`, absent on `TransformerBridge`; no `tokens` or `return_type` parameters |
| Hook calling convention (Q5) | Read `HookPoint.add_hook` source; attached both hook forms to a live `HookPoint` | Called as `hook(activation, hook=self)`; one-argument lambda raised `TypeError: <lambda>() got an unexpected keyword argument 'hook'` |
| Cache tuple key third slot (Q6) | Called `utils.get_act_name` with two and three positional arguments | Both returned `blocks.0.attn.hook_pattern`; the third argument is `layer_type` and was ignored |
| Non-leaf `requires_grad` assignment (Q7) | torch 2.13.0, embeddings from a real `nn.Embedding` forward pass | Raised `RuntimeError: you can only change requires_grad flags of leaf variables.`; `detach()` plus `requires_grad_(True)` populated `.grad` |

Not independently re-verified here, and unchanged from the deck reviews where they were confirmed: the Hydra effect characterization (Q4), grouped-query attention semantics (Q9), and the specific effects of `enable_compatibility_mode()` (Q10). These are conceptual and stable rather than version-sensitive, but the compatibility-mode behavior is worth a confirming glance on the cohort stack if you have already done a cached run.

TransformerLens 3.5.1 pulls transformers 5.x, so it needs an isolated environment. If a student runs any of this code, that constraint applies.
