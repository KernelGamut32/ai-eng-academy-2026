# Week 04 Lab 01 Hints

Three levels per task. Read level 1 first; escalate only if still stuck. Level 3 shows a key line in context but never the whole solution.

---

## Task B1: prev_token_score

**Level 1.** The score for a head is the average of the attention entries that sit one step below the diagonal: positions (q, q - 1) for every query q starting at 1. You want to grab exactly those entries for all heads at once and average them.

**Level 2.** Build an index tensor `q_idx = torch.arange(1, n_query)`. PyTorch advanced indexing lets you pick a diagonal-like set of entries in one shot: indexing the pattern with `[:, q_idx, q_idx - 1]` gives shape `[n_heads, n_query - 1]`. Then reduce over the last dimension. Handle the degenerate `n_query < 2` case by returning zeros.

**Level 3.** The selection line looks like this:

```python
q_idx = torch.arange(1, n_query)
sub_diagonal = pattern[:, q_idx, q_idx - 1]   # [n_heads, n_query - 1]
```

Averaging that over its last dimension is your return value.

---

## Task B2: scan_prev_token

**Level 1.** A plain loop over layers. Each layer's pattern comes out of the cache under the name `get_act_name("pattern", layer)`, and you already wrote the scorer in B1.

**Level 2.** Allocate `torch.zeros(N_LAYERS, N_HEADS)`, loop `for layer in range(N_LAYERS)`, and fill each row with `prev_token_score(...)` of that layer's pattern. Remember the cache entry has a batch dimension in front; index `[0]` to drop it.

**Level 3.** The access line inside the loop:

```python
pattern = cache[get_act_name("pattern", layer)][0]   # [n_heads, q, k]
```

---

## Task C1: logit_lens_topk

**Level 1.** Per layer it is the three-step decode described in the notebook: grab the final-position residual, apply the final LayerNorm, project with the unembedding. Then `torch.topk` and decode each id to a string.

**Level 2.** Skeleton per layer:

1. `resid = cache[get_act_name("resid_post", layer)][0, -1]`
2. lens logits = LayerNorm of resid, matrix-multiplied by `model.W_U`, plus `model.b_U`
3. `top = torch.topk(logits, k)`, then decode `top.indices.tolist()` one id at a time with `model.tokenizer.decode([i])`

Wrap the loop in `torch.no_grad()` since nothing here needs gradients.

**Level 3.** The decode line:

```python
lens_logits = model.ln_final(resid) @ model.W_U + model.b_U
```

---

## Task C2: target_logit_by_layer

**Level 1.** Same decode as C1, but instead of topk you read one entry: the logit at index `target_id`.

**Level 2.** Loop layers, compute the lens logits exactly as in C1, append `float(logits[target_id])` to a list, and return `torch.tensor(that_list)`.

**Level 3.** Inside the loop:

```python
values.append(float(lens_logits[target_id]))
```

---

## Task D1: gradient_x_input

**Level 1.** You need a tensor of embeddings that PyTorch will track gradients for, run the model on it via the provided `forward_with_embeds`, backprop from a single scalar (the target logit), then combine gradient and input.

**Level 2.** Four moves:

1. `embeds = model.embed(tokens).detach().clone().requires_grad_(True)`
2. `logits = forward_with_embeds(tokens, embeds)`
3. `.backward()` on `logits[0, -1, target_id]`
4. multiply `embeds.grad` by `embeds` elementwise, sum over the last (embedding) dimension, take batch row 0, detach

**Level 3.** The final combine:

```python
scores = (embeds.grad * embeds).sum(dim=-1)[0].detach()
```

---

## Task D2: integrated_gradients

**Level 1.** It is D1 inside a loop: at each step you build an interpolated embedding between a zero baseline and the real embeddings, backprop the same scalar, and accumulate the gradient. At the end, average the gradients and multiply by the input difference.

**Level 2.** Skeleton:

1. `actual = model.embed(tokens).detach()`, `baseline = torch.zeros_like(actual)`
2. loop `i in range(steps)` with `alpha = (i + 0.5) / steps` (the midpoint rule; the completeness check depends on it)
3. `interp = (baseline + alpha * (actual - baseline)).detach().clone().requires_grad_(True)`
4. forward, backward, accumulate `interp.grad.detach()` into a running total
5. average by `steps`, then `((actual - baseline) * avg_grad).sum(dim=-1)[0]`

**Level 3.** The two lines people most often get wrong:

```python
alpha = (i + 0.5) / steps
grad_total += interp.grad.detach()
```

A fresh `interp` leaf each iteration means no `zero_grad` bookkeeping is needed.

---

## Task E1: ablate_head

**Level 1.** Write a hook function that receives the layer's `z` tensor (shape `[batch, pos, head, d_head]`), clones it, zeros the one head's slice, and returns it. Run the model once normally and once inside `model.hooks(fwd_hooks=[(name, your_hook)])`, comparing the target's log probability.

**Level 2.** Skeleton:

1. hook: `def zero_head(z, hook): z = z.clone(); z[:, :, head, :] = 0.0; return z`
2. under `torch.no_grad()`: run the model plain, take `torch.log_softmax(logits[0, -1], dim=-1)[target_id]`
3. rerun inside `with model.hooks(fwd_hooks=[(get_act_name("z", layer), zero_head)]):`
4. return the three floats in a dict, delta = ablated minus base

**Level 3.** The hook registration:

```python
with model.hooks(fwd_hooks=[(get_act_name("z", layer), zero_head)]):
    ablated = model(tokens)
```

---

## Task E2: patch_head_z

**Level 1.** Structurally the same hook as E1, except instead of writing zeros into the head's slice you write the clean run's cached values for that same layer and head. Then plug the patched logit into the recovery formula given in the docstring.

**Level 2.** Skeleton:

1. `clean_z = clean_cache[get_act_name("z", layer)]` before defining the hook
2. hook: clone `z`, then `z[:, :, head, :] = clean_z[:, :, head, :]`, return it
3. run the corrupted tokens inside the hooks context, under `torch.no_grad()`
4. `patched_logit = float(patched[0, -1, target_id])`, then apply the recovery formula, guarding against a near-zero denominator

**Level 3.** The patch body:

```python
z = z.clone()
z[:, :, head, :] = clean_z[:, :, head, :]
return z
```

---

## Stretch goals

Level 1 pointers only; full solutions live in the instructor notebook.

**Stretch 1 (Attention x Gradient).** Attention patterns are not leaf tensors, so `.grad` is empty by default. Inside a forward hook, call `retain_grad()` on the activation and stash a reference to it, then backprop the target logit after the forward pass.

**Stretch 2 (Token erasure).** `forward_with_embeds` already lets you run with modified embeddings. Clone the real embeddings, zero one position's row, and compare log probabilities.

**Stretch 3 (Patching scan).** Two nested loops over `range(N_LAYERS)` and `range(N_HEADS)` calling your E2 function, results into a `[n_layers, n_heads]` grid, rendered with `imshow` using a diverging colormap centered at zero.
