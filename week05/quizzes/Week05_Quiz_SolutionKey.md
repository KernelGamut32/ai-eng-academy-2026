# Week 5 Knowledge Check: Solution Key

**AI Engineering Academy | Gamut Technology Services | Instructor-facing. Do not distribute to students.**

Answer distribution: A appears 1 time, B 6 times, C 2 times, D 1 time. The correct answer is not consistently the longest option, and no positional pattern is present. Note the B-weighting and reshuffle if you reuse this set.

Question types: code questions are 2, 3, 5, 7, and 9. Concept questions are 1, 4, 6, 8, and 10.

| Q | Answer | Type | Maps to |
|---|--------|------|---------|
| 1 | B | concept | Module 01, what LoRA trains |
| 2 | A | code | Module 01, adapter parameter count |
| 3 | A | code | Module 01, rank versus alpha |
| 4 | B | concept | Module 02, embedding model consistency |
| 5 | B | code | Module 02, Pinecone namespaces |
| 6 | B | concept | Module 03, grounding and hallucination |
| 7 | B | code | Module 03, retrieval chain structure |
| 8 | A | concept | Module 03, faithfulness versus answer relevancy |
| 9 | B | code | Module 04, config versus log |
| 10 | C | concept | Module 04, artifacts for reproducibility |

---

### Question 1 - Answer: B

LoRA freezes the entire base model and injects a pair of small low-rank matrices into each targeted layer. Only those adapter matrices receive gradient updates. That is the whole point: you get task adaptation while touching a tiny fraction of the parameters, so training fits in far less memory and the resulting artifact is a few megabytes rather than a full model copy.

Why the distractors are wrong. A misreads `lora_alpha` as a learning-rate control on the base weights. It is a scaling factor on the adapter output, and the base weights are not trained at all. C is the most tempting wrong answer, because it correctly senses that only the targeted layers are involved. The error is that even in those layers the original weight matrix is frozen. What trains is the added adapter, not the original layer. D describes something closer to QLoRA's quantized base plus the merge step, but even there the base weights are frozen and only the adapter trains.

---

### Question 2 - Answer: A (code)

A LoRA adapter on a linear layer of shape 2048 by 2048 adds two matrices: A of shape r by 2048 and B of shape 2048 by r, with r equal to 8. That is 8 times 2048 plus 2048 times 8, which is 16,384 plus 16,384, or 32,768 trainable parameters. The full layer has 2048 times 2048, which is 4,194,304 weights. The adapter is 32,768 divided by 4,194,304, about 0.78 percent.

Verified: `r * (d_in + d_out)` equals 32,768 for r equal to 8 and both dimensions 2048, which is 0.78 percent of 4,194,304.

Why the distractors are wrong. C counts only one of the two matrices, forgetting that both A and B are trainable. D doubles by treating the count as r times d_in times d_out rather than r times the sum, which is the natural wrong mental model if you picture the reconstructed update matrix instead of the two factors. B is the deepest trap: the reconstructed update B times A does have the same 2048 by 2048 shape as the original layer, so it is easy to conclude the parameter count matches. It does not, because you store and train the two thin factors, not their product. That distinction is exactly why LoRA saves memory.

---

### Question 3 - Answer: A (code)

Rank r sets the adapter's capacity, the dimensionality of the low-rank update. Config B doubles r from 8 to 16, so it has twice the adapter capacity and twice the trainable parameters. Separately, the adapter output is scaled by `lora_alpha / r`. For A that is 16 over 8, which is 2.0. For B that is 16 over 16, which is 1.0. So B has more capacity but a smaller scaling factor.

Verified: alpha over r gives 2.0 for config A and 1.0 for config B.

Why the distractors are wrong. B inverts the relationship, claiming higher rank compresses the update. Higher rank increases capacity. C treats equal `lora_alpha` as making the configs equivalent, ignoring that r differs, which changes both capacity and the scaling denominator. D gets the scaling backwards, assigning 2.0 to config B. It is config A that scales by 2.0. The two levers being independent, capacity from r and scaling from alpha over r, is the point the question tests, and it is why a common tuning recipe is to set alpha to a fixed multiple of r so the scaling stays constant as you sweep rank.

---

### Question 4 - Answer: B

An embedding model defines its own coordinate space. Two different models can both output, say, 384-dimensional vectors while placing the same sentence in completely different locations. Cosine similarity across those two spaces is meaningless, so the nearest neighbors the index returns are effectively random with respect to meaning. Nothing errors, because the only structural constraint the index enforces is dimension, and the dimension matches. This is the silent-failure case: the pipeline runs, returns results, and quietly retrieves the wrong things. The rule that follows is that the query path must use the identical embedding model that built the index.

Why the distractors are wrong. A states the misconception directly. Same dimension is necessary but nowhere near sufficient. C invents automatic re-embedding, which the index does not do. It stores and compares raw vectors. D assumes an error that does not occur, which is the crux: the danger is precisely that there is no error to alert the teammate.

---

### Question 5 - Answer: B (code)

The upsert targets `namespace="catalog"`, but the query omits the namespace argument. A Pinecone query with no namespace searches the default namespace, not all namespaces, and the default namespace does not contain sku-1001. So the query runs, filters, and returns zero matches, all without error. Add `namespace="catalog"` to the query and the match appears.

Why the distractors are wrong. C blames the metadata filter, which is well formed here and matches the upserted metadata. It is the plausible wrong answer for anyone who did not notice the missing namespace, since a filter is the other thing that can silently zero out results. A misunderstands `top_k`, which caps how many neighbors return, not whether a clearly relevant vector can be found at all. D invents nondeterminism in `embed()`. For a fixed model and input the embedding is stable, and even if it drifted slightly, cosine similarity would still rank the same item near the top. The namespace mismatch is the mainline Pinecone gotcha this question targets.

---

### Question 6 - Answer: B

Retrieval grounds the model by putting relevant source text in the context, which sharply reduces hallucination. It does not eliminate it. The model can still assert things the context does not support, blend in parametric knowledge, or over-generalize. That is exactly why the week measures faithfulness as its own metric: faithfulness asks whether each claim in the answer is actually supported by the retrieved context, independent of whether the answer sounds right.

Why the distractors are wrong. A and D both overclaim that retrieval makes hallucination impossible, differing only in D's added condition. Neither is true. The model is still generating, not quoting. C swings to the opposite error, denying that retrieval helps at all. Retrieval is the single most effective grounding lever in the pipeline. The correct position is the middle one: a large reduction, not a guarantee.

---

### Question 7 - Answer: B (code)

`create_retrieval_chain` wires two stages together. The retriever fetches the relevant chunks, and `create_stuff_documents_chain` is the generation stage: it stuffs those already-retrieved documents into the prompt as context and hands the assembled prompt to the LLM to answer from. The name refers to the stuffing strategy, placing the document text directly into a single prompt. So in this pipeline the retriever finds context and the stuff-documents chain consumes it.

Why the distractors are wrong. A assigns retrieval to the wrong component. The retriever passed as the first argument to `create_retrieval_chain` does that. D assigns embedding and similarity search to it, which is also the retriever's job, done through the vector store. C invents a rerank-and-discard step. Stuffing passes the retrieved documents through into the prompt. It does not prune them to one. The discriminator is understanding that this chain is the generation half, not the retrieval half, of the two-stage flow.

---

### Question 8 - Answer: A

Answer relevancy asks a narrow question: does the answer address what was asked. A confident, fluent, entirely invented answer to an out-of-corpus question can be highly relevant to the question while being completely unsupported by any retrieved context. Raising the relevancy threshold therefore does not catch invention. What catches it is faithfulness, which checks whether the answer's claims are grounded in the retrieved context, paired with an abstention behavior that lets the system decline when the context does not contain the answer. The two metrics measure different things, and this failure lives in the gap between them.

Why the distractors are wrong. B assumes relevancy already covers grounding, which is the exact conflation the question is built to expose. C claims the two metrics are identical, which is false. They can move in opposite directions, and this scenario is a case where relevancy is high and faithfulness is low. D overstates the remedy, claiming only retraining can help, when the practical fixes are an abstention instruction and a faithfulness gate. Naming the faithfulness-versus-relevancy distinction is the learning objective here.

---

### Question 9 - Answer: B (code)

`wandb.config` captures the run's fixed inputs, the hyperparameters you set once and want attached to the run for later filtering and comparison across runs. `wandb.log` records values that evolve during the run, such as the per-step training loss and the final evaluation F1. They serve different roles, which is why the loss goes through `log` inside the loop and the rank and learning rate go into `config` at init.

Why the distractors are wrong. A claims they are interchangeable. You can technically log a constant, but config is the right home for fixed hyperparameters, and conflating them defeats cross-run comparison. C invents a one-call limit on `wandb.log`. It is called many times per run, once or more per step. D claims `step` cannot be passed, when `step` is a supported argument to `wandb.log` and is the normal way to place a metric on the x-axis. The config-versus-log split is the mainline W&B concept this tests.

---

### Question 10 - Answer: C

`wandb.Artifact` is the versioning mechanism. It stores and versions the actual files, the dataset snapshot and the model or adapter output, and links each run to the specific artifact versions it consumed and produced. That lineage is what makes a three-week-old fine-tune reproducible: you can retrieve the exact data and base artifacts that run used, rather than hoping the files on disk are unchanged.

Why the distractors are wrong. A and B both name things that record information about a run but not the run's data and model files. Logged metrics and config values tell you what happened and with which settings, but they do not preserve the bytes of the dataset or the model. B is the more tempting of the two, because config does capture the hyperparameters, which feels like reproducibility until you realize the training data snapshot and the output adapter are not in it. D names the system metrics panel, which records hardware and utilization, useful for performance debugging and irrelevant to reproducing the data and model versions.

---

## Scoring and use

Suggested cut line for solid understanding is 7 of 10. The material is broad this week, spanning four tools, so expect the spread to track which modules a given student engaged with most.

The questions most likely to separate the room are 2 (the reconstructed-update-shape trap in the parameter count), 4 (the same-dimension silent failure), and 8 (faithfulness versus answer relevancy). Question 4 is the most valuable, because the silent-failure instinct it builds carries directly into production RAG debugging.

Fast debrief order if time is short: 4, 8, 2, 6, then the rest. Those four carry the durable lessons, which are that embedding spaces are model-specific, that grounding is measured not assumed, that LoRA stores factors rather than the full update, and that retrieval reduces rather than removes hallucination.

---

## Verification ledger

Code-bearing arithmetic was computed rather than recalled.

| Claim | How verified | Result |
|---|---|---|
| LoRA adapter parameter count (Q2) | Computed `r * (d_in + d_out)` for r=8, dims 2048 | 32,768 trainable, 0.78 percent of 4,194,304 |
| alpha over r scaling (Q3) | Computed alpha divided by r for both configs | Config A scales 2.0, config B scales 1.0; B has double the rank |
| Cosine similarity behavior (Q4) | Computed cosine on same-model related and unrelated vectors | Related 0.99, unrelated 0.16; distinction holds only within one model's space |

Not re-verified here and unchanged from the deck reviews where they were confirmed: Pinecone default-namespace behavior on a query with no namespace (Q5), the `create_retrieval_chain` and `create_stuff_documents_chain` roles and the `langchain_classic` import path (Q7), and the W&B `config` versus `log` versus `Artifact` roles (Q9, Q10). These are stable API and conceptual facts. The `langchain_classic` import path in Q7 is version-sensitive and was confirmed during the Week 5 Lab 03 build against langchain 1.x. Confirm it against the exact pin on the cohort image before administering if that pin has moved.
