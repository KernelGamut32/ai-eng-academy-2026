# Week 04 Lab 01 Detailed Hints

This is the heavier tier. Where `HINTS.md` guides the approach in three escalating levels, this file shows the working core of each task with a line-by-line explanation. Use it when you want to move fast or when the progressive hints were not enough. What is left to you is the function shell, the glue, and understanding why each line is there. The instructor solution notebook remains the only place with fully assembled, executed functions.

Pick one file per task. Reading both hint tiers for the same task wastes your time.

---

## Task B1: prev_token_score

The whole task is selecting the sub-diagonal of every head's attention matrix and averaging it.

```python
n_query = pattern.shape[-2]          # number of query positions
if n_query < 2:                      # a 1-token sequence has no "previous token"
    return torch.zeros(pattern.shape[0])

q_idx = torch.arange(1, n_query)     # queries 1..end; query 0 has no predecessor
sub = pattern[:, q_idx, q_idx - 1]   # entries (1,0), (2,1), (3,2), ... for every head
```

The indexing line is the entire trick. Two aligned index tensors walk in lockstep, so `[q_idx, q_idx - 1]` picks exactly the "look one step back" cell for each query, for all heads at once. `sub` has shape `[n_heads, n_query - 1]`.

Your return value is the mean of `sub` over its last dimension: one score per head. A perfect previous-token head averages 1.0.

---

## Task B2: scan_prev_token

A fill-the-grid loop reusing B1.

```python
scores = torch.zeros(N_LAYERS, N_HEADS)
for layer in range(N_LAYERS):
    pattern = cache[get_act_name("pattern", layer)][0]   # [0] drops the batch dim
    scores[layer] = prev_token_score(pattern)
```

The only trap is forgetting the `[0]`: the cache stores `[batch, head, q, k]` and B1 expects `[head, q, k]`. Return `scores`.

---

## Task C1: logit_lens_topk

Per layer, the model's own output head is applied early: residual, then final LayerNorm, then unembedding.

```python
with torch.no_grad():                                    # pure reading, no gradients
    for layer in range(N_LAYERS):
        resid = cache[get_act_name("resid_post", layer)][0, -1]   # final position, [d_model]
        lens_logits = model.ln_final(resid) @ model.W_U + model.b_U
        top = torch.topk(lens_logits, k)
        strings = [model.tokenizer.decode([i]) for i in top.indices.tolist()]
```

- `[0, -1]` selects batch 0, last position: the slot whose next-token prediction we care about.
- `model.ln_final(...) @ model.W_U + model.b_U` is literally the model's output path. Skipping `ln_final` gives systematically wrong logits, which is the most common bug here.
- `decode([i])` takes a list because the tokenizer decodes sequences; one id still needs brackets.

Collect `(layer, strings, values)` tuples into a list and return it. Convert the topk values to plain floats when you store them.

---

## Task C2: target_logit_by_layer

Identical decode to C1; read one entry instead of topk.

```python
values = []
with torch.no_grad():
    for layer in range(N_LAYERS):
        resid = cache[get_act_name("resid_post", layer)][0, -1]
        lens_logits = model.ln_final(resid) @ model.W_U + model.b_U
        values.append(float(lens_logits[target_id]))
```

Return `torch.tensor(values)`: one number per layer, the belief curve for the target token.

---

## Task D1: gradient_x_input

Four moves: make a gradient-tracked copy of the embeddings, run the model on it, backprop from one scalar, combine.

```python
embeds = model.embed(tokens).detach().clone().requires_grad_(True)
logits = forward_with_embeds(tokens, embeds)
logits[0, -1, target_id].backward()
scores = (embeds.grad * embeds).sum(dim=-1)[0].detach()
```

- `detach().clone().requires_grad_(True)` builds a fresh leaf tensor: detached from the model's graph, our own copy, marked so PyTorch delivers gradients into `embeds.grad`.
- `logits[0, -1, target_id]` is a scalar, which is what `backward()` needs as a starting point.
- `embeds.grad * embeds` weights sensitivity by what is actually present; summing over the last dimension collapses 768 embedding numbers into one score per token. `[0]` drops the batch dimension.

Because `embeds` is created fresh inside your function, its `.grad` starts empty and no `zero_grad` bookkeeping is needed. Return `scores`, shape `[seq_len]`, one entry per token including BOS at index 0.

---

## Task D2: integrated_gradients

D1 inside a loop over a straight path from a zero baseline to the real embeddings.

```python
actual = model.embed(tokens).detach()
baseline = torch.zeros_like(actual)
grad_total = torch.zeros_like(actual)

for i in range(steps):
    alpha = (i + 0.5) / steps                    # midpoint rule: slice centers, not endpoints
    interp = (baseline + alpha * (actual - baseline)).detach().clone().requires_grad_(True)
    logits = forward_with_embeds(tokens, interp)
    logits[0, -1, target_id].backward()
    grad_total += interp.grad.detach()

avg_grad = grad_total / steps
scores = ((actual - baseline) * avg_grad).sum(dim=-1)[0]
```

- The `(i + 0.5) / steps` alpha is what makes the completeness check pass tightly. Endpoint spacing (`linspace(0, 1, steps)`) leaves a visible integration error; the check will catch it.
- A fresh `interp` leaf each iteration means gradients cannot leak between steps.
- The final line is average slope times distance traveled, per coordinate, summed to per-token scores.

Return `scores`. Expect the cell to take about a minute on CPU at 64 steps; that is normal, not a hang.

---

## Task E1: ablate_head

A hook that silences one head, then a before-and-after comparison in log probability.

```python
def zero_head(z, hook):
    z = z.clone()                 # never mutate the tensor the model handed you
    z[:, :, head, :] = 0.0        # all batches, all positions, this head, all dims
    return z

with torch.no_grad():
    base = model(tokens)
    base_lp = float(torch.log_softmax(base[0, -1], dim=-1)[target_id])
    with model.hooks(fwd_hooks=[(get_act_name("z", layer), zero_head)]):
        ablated = model(tokens)
    abl_lp = float(torch.log_softmax(ablated[0, -1], dim=-1)[target_id])
```

- The hook point `get_act_name("z", layer)` resolves to `blocks.L.attn.hook_z`, shape `[batch, pos, head, d_head]`: the one place a single head's output is still a separable slice.
- Log probability rather than raw logit because softmax is competitive; zeroing a head shifts every logit, and log probs account for the whole field.

Assemble the return dict with `base_logprob`, `ablated_logprob`, and `delta_logprob` computed as ablated minus base, so negative means the head was helping.

---

## Task E2: patch_head_z

Same hook shape as E1, but you transplant the clean run's cached values instead of zeros.

```python
clean_z = clean_cache[get_act_name("z", layer)]   # fetch once, outside the hook

def patch(z, hook):
    z = z.clone()
    z[:, :, head, :] = clean_z[:, :, head, :]
    return z

with torch.no_grad():
    with model.hooks(fwd_hooks=[(get_act_name("z", layer), patch)]):
        patched = model(corrupt_tokens)
patched_logit = float(patched[0, -1, target_id])
```

The run is the corrupted prompt end to end; exactly one head lives in the clean world. For the recovery fraction:

```python
denom = clean_logit - corrupt_logit
recovery = (patched_logit - corrupt_logit) / denom if abs(denom) > 1e-6 else float("nan")
```

The guard avoids manufacturing a huge, meaningless recovery number when the clean and corrupt worlds barely differ. Return the dict with `patched_logit` and `recovery`. Values outside 0 to 1 are real results, not bugs; they mean the head interacts with others.

---

## Stretch goals

Detailed cores. Full assembled versions remain instructor-only.

**Stretch 1 (Attention x Gradient).** Patterns are intermediate tensors, so PyTorch discards their gradients unless asked to keep them:

```python
stash = {}

def keep(act, hook):
    act.retain_grad()             # ask PyTorch to keep this tensor's gradient
    stash[hook.name] = act        # reference must outlive the hook
    return act
```

Run the forward pass with this hook on the pattern point, backprop the target logit, then `stash[name] * stash[name].grad` weights each attention edge by how much the output cares about it. Sum over heads and query positions to get one score per key token.

**Stretch 2 (Token erasure).** Clone the real embeddings, zero one position's row, rerun through `forward_with_embeds`, and compare target log probabilities:

```python
erased = model.embed(tokens).detach().clone()
erased[0, pos, :] = 0.0
```

When choosing which position to erase from your D1 scores, skip BOS: argmax over `scores[1:]` and add 1 back to the index.

**Stretch 3 (Patching scan).** Two nested loops calling your E2 function:

```python
grid = torch.zeros(N_LAYERS, N_HEADS)
for layer in range(N_LAYERS):
    for head in range(N_HEADS):
        grid[layer, head] = patch_head_z(corrupt_tokens, layer, head, clean_cache,
                                         TARGET_E, CLEAN_LOGIT, CORRUPT_LOGIT)["recovery"]
```

Render with `imshow` using a diverging colormap (`"RdBu_r"`) centered at zero so positive and negative recovery read at a glance. About 144 forward passes, roughly a minute on CPU.
