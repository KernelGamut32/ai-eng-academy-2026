# Week 5 Knowledge Check: Fine-Tuning and Retrieval-Augmented Generation

**AI Engineering Academy | Gamut Technology Services**

Ten multiple choice questions covering this week's material (Modules 01 through 04): adapter-based fine-tuning with LoRA, embeddings and Pinecone index management, RAG pipeline construction and evaluation, and experiment tracking with Weights and Biases.

**Instructions.** Choose the single best answer for each question. Five of the ten show code and ask you to reason about what it does. Read the code carefully. Some distractors are true statements pointed at the wrong setting, so match the claim to the exact situation shown. The questions stay on the core ideas of the week, not edge cases.

Closed book. Approximately 25 minutes.

---

### Question 1

You fine-tune a 360M-parameter instruct model with LoRA. During training, which parameters actually receive gradient updates?

- A. All of the base model weights, but at a reduced learning rate set by `lora_alpha`.
- B. Only the low-rank adapter matrices injected into the targeted layers. The base model weights stay frozen.
- C. The base model weights in the targeted layers only. The other layers are frozen.
- D. A quantized copy of the full model, which is then merged back after training.

---

### Question 2

This is the adapter configuration from the lab.

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

For one targeted linear layer of shape 2048 by 2048, the adapter adds two matrices and freezes the original. How many trainable parameters does the adapter add for that one layer, and how does that compare to training the full layer?

- A. 32,768, which is about 0.78 percent of the layer's 4,194,304 weights.
- B. 4,194,304, the same as the full layer, because the update matrix has the same shape as the original.
- C. 16,384, because only one matrix of shape 8 by 2048 is trainable.
- D. 65,536, because the rank multiplies both dimensions.

---

### Question 3

Two engineers configure LoRA differently for the same base layer.

```python
config_a = LoraConfig(r=8,  lora_alpha=16, target_modules=["q_proj", "v_proj"])
config_b = LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"])
```

The adapter's contribution to the layer is scaled by `lora_alpha / r`. Which statement correctly describes the difference between these two configs?

- A. Config B has twice the adapter capacity of A, and its adapter output is scaled by 1.0 rather than A's 2.0.
- B. Config B has half the adapter capacity of A, because a higher rank compresses the update.
- C. The two configs are equivalent, because `lora_alpha` is identical.
- D. Config B scales its adapter output by 2.0, which doubles the effective learning rate.

---

### Question 4

A vector index was built by embedding a product catalog with one model. A teammate later writes the query path with a different embedding model that happens to produce vectors of the same dimension. The index accepts the query and returns results with no error. What is actually happening?

- A. Nothing is wrong. Same dimension means the vectors are interchangeable across models.
- B. The query is silently returning near-random neighbors, because two models place text in different coordinate spaces even at the same dimension, and cosine similarity between them is not meaningful.
- C. The index automatically re-embeds the query with the original model, so results are correct.
- D. The query raises a dimension-mismatch error that the teammate is catching and ignoring.

---

### Question 5

This snippet upserts and queries a Pinecone index.

```python
index.upsert(
    vectors=[("sku-1001", embed("cordless drill"), {"category": "power_tools"})],
    namespace="catalog",
)

results = index.query(
    vector=embed("cordless drill"),
    top_k=3,
    filter={"category": "power_tools"},
)
```

The upsert succeeds, but the query returns zero matches even though sku-1001 is clearly relevant. What is the most likely cause given this code?

- A. `top_k=3` is too small to return the vector.
- B. The query omits `namespace="catalog"`, so it searches the default namespace, which does not contain the upserted vector.
- C. The metadata filter is malformed and silently excludes everything.
- D. `embed()` returns a different vector for the same string on the second call, so the match fails.

---

### Question 6

In a RAG pipeline, a colleague argues that grounding answers in retrieved context "makes hallucination impossible." Based on this week's evaluation material, what is the most accurate response?

- A. Correct. Once context is retrieved, the model can only quote from it.
- B. Retrieval reduces hallucination by grounding the answer, but the model can still produce claims not supported by the retrieved context, which is why faithfulness is measured separately.
- C. Incorrect. Retrieval has no effect on hallucination and only improves latency.
- D. Correct, as long as the retrieval step returns at least one document.

---

### Question 7

This is the core of the retrieval chain from the RAG lab.

```python
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

combine_docs = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs)

response = rag_chain.invoke({"input": "What is the return window for power tools?"})
```

What does `create_stuff_documents_chain` do in this pipeline?

- A. It retrieves the documents from the vector store based on the input query.
- B. It takes the documents the retriever already returned and inserts them into the prompt as context for the LLM to answer from.
- C. It ranks the retrieved documents and discards all but the single best match.
- D. It embeds the input query so the retriever can perform the similarity search.

---

### Question 8

Your RAG system answers grounded questions well, but on questions the corpus does not cover, it confidently invents an answer instead of declining. Two evaluation metrics are on the table: faithfulness and answer relevancy. A teammate proposes raising the answer-relevancy threshold to fix this. Why does that not address the problem?

- A. Answer relevancy measures whether the answer addresses the question, not whether it is grounded in the retrieved context, so a fluent invented answer can score high on it. Faithfulness and an abstention check are what catch unsupported answers.
- B. Answer relevancy already prevents invented answers, so the teammate's fix will work.
- C. Faithfulness and answer relevancy are the same metric under different names, so changing either has no effect.
- D. The only fix is to retrain the base model, since evaluation metrics cannot influence this behavior.

---

### Question 9

This loop logs metrics during fine-tuning with Weights and Biases.

```python
import wandb

wandb.init(project="cordwell-lora", config={"r": 8, "lora_alpha": 16, "lr": 2e-4})

for step, batch in enumerate(train_loader):
    loss = train_step(batch)
    wandb.log({"train/loss": loss}, step=step)

wandb.log({"eval/f1": final_f1})
wandb.finish()
```

Which statement about this code is correct?

- A. `config` and `wandb.log` are interchangeable, so the hyperparameters could equally have been logged with `wandb.log`.
- B. `config` records the run's fixed hyperparameters for later comparison across runs, while `wandb.log` records values that change over the course of the run, such as the per-step loss.
- C. `wandb.log` must be called exactly once per run, so logging both loss and F1 will raise an error.
- D. The per-step loss will not appear in the dashboard, because `step` is reserved and cannot be passed to `wandb.log`.

---

### Question 10

Your team wants to reproduce a fine-tune from three weeks ago: same base model, same training data snapshot, same adapter output. Within Weights and Biases, which mechanism is designed to make the exact data and model versions reproducible, as opposed to just recording metrics?

- A. `wandb.log`, because every logged metric is timestamped.
- B. `wandb.config`, because it stores the learning rate and rank.
- C. `wandb.Artifact`, because it versions and stores the actual dataset and model files, linking each run to the exact inputs and outputs it used.
- D. The run's system metrics panel, because it captures the hardware the run used.

---

*End of quiz. Ten questions. Answer key is a separate file.*
