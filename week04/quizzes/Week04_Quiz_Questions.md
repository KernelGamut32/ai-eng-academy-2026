# Week 4 Knowledge Check: LLM Behavior Analysis and Evaluation Techniques

**AI Engineering Academy | Gamut Technology Services**

Ten multiple choice questions covering this week's material (Modules 00, 01, and 02): transformer internals and the residual stream, attention and specialized heads, logits and sampling, and the TransformerLens interpretability workflow including caching, hooks, ablation, patching, and gradient-based saliency.

**Instructions.** Choose the single best answer for each question. Five of the ten show code and ask you to reason about what it does. Read the code carefully. Several distractors are true statements about a different object, a different API version, or a different model family, so match the claim to the exact situation shown. In at least one question the code runs without raising anything and is still wrong.

Closed book. Approximately 25 minutes.

---

### Question 1

A student states that in a transformer, every token attends to every other token in the sequence. For the decoder-only models this week's material centers on, such as GPT-2 and Llama, what is the most accurate correction?

- A. The statement is correct. Causal masking is applied during training only and is removed at inference.
- B. Each token attends only to itself and to earlier positions, because a causal mask blocks attention to future positions.
- C. Each token attends only to the tokens inside its own attention head, never across heads.
- D. Each token attends only to the immediately preceding token, which is why position embeddings are required.

---

### Question 2

The following runs cleanly.

```python
import numpy as np

logits = np.array([8.2, 7.1, 2.3, -1.5])   # cat, dog, bird, the

def softmax(z, T):
    z = z / T
    e = np.exp(z - z.max())
    return e / e.sum()

for T in (0.5, 1.0, 2.0):
    p = softmax(logits, T)
    print(f"T={T}: {np.round(p, 3)}  argmax={p.argmax()}")
```

Output:

```
T=0.5: [0.9   0.1   0.     0.   ]  argmax=0
T=1.0: [0.749 0.249 0.002 0.   ]  argmax=0
T=2.0: [0.611 0.352 0.032 0.005]  argmax=0
```

What does this output establish about temperature?

- A. Temperature reshapes the distribution but never changes which token holds the highest probability, so greedy decoding is unaffected by it.
- B. Temperature above 1.0 will eventually make a lower-ranked token the most probable, which is the source of creative output.
- C. Temperature changes the ranking of tokens, which is why higher temperature produces different greedy output.
- D. Temperature is applied after the softmax, so it rescales probabilities without touching the logits.

---

### Question 3

A student adapts a logit attribution example and runs this against a current TransformerLens install.

```python
from transformer_lens.model_bridge import TransformerBridge

model = TransformerBridge.boot_transformers("openai-community/gpt2", device="cpu")
model.enable_compatibility_mode()

tokens = model.to_tokens("The cat sat on the")
contributions = model.accumulated_resid(tokens, return_type="logits")
```

What happens on the last line?

- A. It returns per-component logit contributions, which is the intended behavior.
- B. It raises `AttributeError`, because `accumulated_resid` is a method on the activation cache returned by `run_with_cache`, not on the model.
- C. It runs but returns contributions for only the final layer, because no layer argument was supplied.
- D. It raises `TypeError`, because `return_type` must be `"attribution"` rather than `"logits"`.

---

### Question 4

Your ablation study zeroes a single attention head that scored highest on an induction metric. Model loss on the task rises only slightly. A teammate concludes the head was not actually important. Why is that conclusion premature?

- A. Zero ablation always produces a small loss change, so the magnitude carries no information.
- B. Loss is the wrong metric. Only KL divergence against the clean distribution can detect a causal effect.
- C. Other components can compensate for the removed head, an effect named the Hydra effect, so a small loss change understates how much that head contributed in the intact model.
- D. Ablation measures correlation rather than causation, so it cannot support any claim about importance.

---

### Question 5

This hook is meant to zero the output of head 7 in layer 3 during a forward pass.

```python
def run_ablated(model, tokens):
    return model.run_with_hooks(
        tokens,
        fwd_hooks=[("blocks.3.attn.hook_z", lambda act: act[:, :, 7, :].zero_())],
    )
```

What is the outcome when this runs?

- A. It works as intended and returns logits from the ablated forward pass.
- B. It raises `TypeError`, because TransformerLens calls the hook with the activation plus a `hook` keyword argument, so a one-argument lambda cannot accept it.
- C. It raises `KeyError`, because `hook_z` is not a valid hook name on the bridge.
- D. It runs but ablates every head, because slicing a tensor inside a lambda drops the head dimension.

---

### Question 6

A student wants the attention pattern for head 3 of layer 0 and writes this. It runs with no error and no warning.

```python
logits, cache = model.run_with_cache(tokens)

head_3_pattern = cache["pattern", 0, 3]
print(head_3_pattern.shape)
```

What is wrong?

- A. Nothing. This is the documented way to select a single head from the cache.
- B. The third element of a tuple cache key is the layer type, not a head index, so the `3` is ignored and this returns the pattern for all heads in layer 0.
- C. It returns head 3 of layer 3, because the layer and head arguments are transposed.
- D. It returns an empty tensor, because head indices in the cache are zero-based and must be passed as a slice.

---

### Question 7

This is the interpolation loop from an integrated gradients implementation. `actual` comes from the model's embedding layer with gradients enabled.

```python
baseline = torch.zeros_like(actual)

for alpha in torch.linspace(0, 1, steps):
    interpolated = baseline + alpha * (actual - baseline)
    interpolated.requires_grad = True
    logits = model(interpolated, start_at_layer=0)
    logits[0, -1, target_id].backward()
    grads.append(interpolated.grad)
```

What is the defect in the two lines that create `interpolated` and set `requires_grad`?

- A. `torch.zeros_like` produces a baseline that requires grad, so the subtraction breaks the graph.
- B. `interpolated` is the result of an arithmetic operation on a tensor that carries grad history, so it is a non-leaf tensor. Assigning `requires_grad` on a non-leaf raises `RuntimeError`. It must be detached first, then marked with `requires_grad_(True)`.
- C. `requires_grad` must be set before the tensor is created, so the assignment is a no-op and `.grad` is silently `None`.
- D. `linspace(0, 1, steps)` starts at zero, so the first iteration produces an all-zero tensor that cannot carry a gradient.

---

### Question 8

You visualize attention and find that in a summarization prompt, one head places 0.87 of its attention weight on a single source sentence. A stakeholder asks whether this proves the model based its summary on that sentence. What is the most defensible answer?

- A. Yes. Attention weights are the model's own accounting of what it used, so this is direct evidence.
- B. Yes, provided the weight exceeds 0.5, which is the accepted significance threshold for attention attribution.
- C. No. Attention weight is a correlational signal about where a head looked. Establishing that the component mattered requires an intervention such as ablation or activation patching that measures the effect on the output.
- D. No. Attention weights are internal scores that have no relationship to which tokens influence the output.

---

### Question 9

A model card states that the architecture uses 32 query heads and 8 key-value heads. What does this design buy, and what is the tradeoff?

- A. It runs 8 attention heads instead of 32, cutting attention compute by a factor of four at some cost to representational variety.
- B. Groups of query heads share one key-value projection, which shrinks the key-value cache by a factor of four at inference. The number of query heads is unchanged, and the cost is some loss of head-specific key-value diversity.
- C. It processes keys and values at one quarter of the sequence length, reducing memory at the cost of dropping the oldest context.
- D. It splits attention across four devices, with 8 heads per device, at the cost of added communication overhead.

---

### Question 10

The Module 02 workflow boots a model through `TransformerBridge.boot_transformers(...)` and then calls `model.enable_compatibility_mode()` before any analysis. What is the consequence of skipping that second call?

- A. The model will not load at all, since compatibility mode completes the boot sequence.
- B. Only performance is affected. The bridge runs more slowly without it, and all numerical results are identical.
- C. The bridge keeps raw HuggingFace weights and naming, so legacy hook aliases may not resolve and the folded-LayerNorm and centered-weight numerics the interpretability recipes assume are absent. Results will differ from what the recipes expect.
- D. It only affects models other than GPT-2, since GPT-2 is the reference architecture the bridge was built against.

---

*End of quiz. Ten questions. Answer key is a separate file.*
