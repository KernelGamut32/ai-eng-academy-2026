# Module 05 Knowledge Check: Reference-Free Evaluation with TruLens

**AI Engineering Academy | Gamut Technology Services**

Five multiple choice questions covering Module 05: why reference-based metrics stop working on open-ended output, the RAG Triad as a diagnostic tool, instrumenting a LangChain app with TruLens 2.x, logging eval runs to MLflow, and designing an eval strategy from CI to production.

All questions use the Cordwell Home and Hardware scenario from the module: a store-associate product Q&A assistant built as a RAG pipeline over a product-location corpus.

**Instructions.** Choose the single best answer for each question. Three of the five show code and ask you to reason about what it does. Read the code carefully. In one question the code runs and raises nothing while quietly recording results under the wrong identity.

Closed book. Approximately 12 minutes.

---

### Question 1

The module's hallucination example pairs a reference with a model response that appends a fabricated claim.

**Reference:** "Use CordSeal S-100 shellac primer on pine before oil paint."

**Model response:** "Use CordSeal S-100 shellac primer on pine before oil paint. Cordwell Pro members also receive a 40% contractor discount on all primers."

The discount does not exist. Scored with ROUGE against that reference, the response earns a recall of 1.0000 on ROUGE-1, ROUGE-2, and ROUGE-L. Why, and what would actually catch this?

- A. ROUGE is miscalibrated on short texts. Scoring longer passages, or switching from ROUGE-1 to ROUGE-L, would surface the fabrication.
- B. Supplying several additional reference answers would let ROUGE penalize the discount claim as unsupported.
- C. Every reference n-gram appears in the response, so recall is perfect by construction. The fabrication is additive text, and a recall-oriented overlap metric has no mechanism to penalize content the reference never mentioned. Catching it requires a reference-free check that traces each claim back to the retrieved source, which is what groundedness measures.
- D. Fabrications of this kind cannot be detected automatically. Only human review of sampled outputs will find them.

---

### Question 2

Your Cordwell assistant returns these mean Triad scores over the seeded query set:

| Leg | Score |
|---|---|
| Context Relevance | 0.91 |
| Groundedness | 0.34 |
| Answer Relevance | 0.88 |

Which component is broken, and how do you know?

- A. The retriever. A groundedness score that low means the chunks coming back do not match the query, so the fix is the embedding model, chunk size, or top-k.
- B. The generator. High context relevance says retrieval delivered chunks that match the query, and high answer relevance says the response addresses the question. Low groundedness says the response asserts claims those retrieved chunks do not support, so the fabrication is happening in the LLM rather than in retrieval.
- C. The prompt framing. The response is drifting off-topic relative to what the user asked, which is what a low score on any single leg indicates.
- D. Nothing is conclusively broken. Two of the three legs are strong, so the mean across the Triad is acceptable and this is within normal variance.

---

### Question 3

A teammate wires up the recorder for the Cordwell app. It runs and raises nothing.

```python
tru_recorder = TruChain(
    chain,
    app_id="cordwell_qa",
    feedbacks=[m_groundedness, m_answer_rel],
)

with tru_recorder as recording:
    for q in queries:
        tru_recorder.app.invoke(q)
```

Later, `session.get_records_and_feedback(app_name="cordwell_qa", app_version="v1")` returns no rows, even though the queries ran. What happened?

- A. Nothing is wrong with the constructor. The empty result means feedback functions failed silently and no records were written at all.
- B. `TruChain` rejects `app_id` as an unknown keyword, so the recorder was never constructed and the `with` block was a no-op.
- C. `app_id` was accepted and stored verbatim as the application's identifier, but because `app_version` was never supplied the records cannot be grouped and the query returns nothing.
- D. `app_id` is a computed hash, not a constructor argument. Passing it silently populates the app **name** instead, `app_version` falls back to its default rather than `"v1"`, and the real `app_id` becomes a hash. The records exist under a different app identity than the one being queried.

---

### Question 4

This is the CI gate from the module, with one line removed.

```python
with tru_recorder as recording:
    for q in SEED_QUERIES:
        tru_recorder.app.invoke(q)

df, _ = session.get_records_and_feedback(
    app_name="cordwell_qa", app_version=VERSION)

assert df["Groundedness"].mean() >= 0.80, "Groundedness gate failed"
```

The build breaks on this step even though the app is behaving normally. What was removed and why does its absence break the gate?

- A. Nothing was removed. Exiting the `with` block blocks until every feedback function has finished, so any failure here is a real quality regression.
- B. The loop draining `record.retrieve_feedback_results()` over `recording.records` was removed. Feedback is computed asynchronously, so at this point the scores have not landed and the metric columns are not yet available on the returned DataFrame. The gate fails on a measurement artifact rather than on model quality.
- C. A cache-clearing call was removed. Without it the DataFrame holds scores carried over from the previous CI run, so the gate evaluates the wrong build.
- D. `get_records_and_feedback` returns only a DataFrame, so unpacking it into two names raises a `ValueError` before the assertion is ever reached.

---

### Question 5

This is the MLflow logging block from the module.

```python
with mlflow.start_run(run_name="cordwell_qa_v1"):
    mlflow.log_params({
        "app_version": "v1",
        "retrieval_k": 3,
        "backend_mode": BACKEND_MODE,      # offline | ollama | lmstudio
    })
    mlflow.log_metric("groundedness_mean", df["Groundedness"].mean())
    mlflow.log_metric("answer_relevance_mean", df["Answer Relevance"].mean())
```

Why is `backend_mode` logged as a run parameter alongside the app version and retrieval settings?

- A. MLflow uses it to route the run to the correct experiment, so metrics from offline and live runs land in separate comparison views.
- B. It captures the latency profile of the run, since a local model server responds more slowly than the deterministic offline path.
- C. The judge is part of the measuring instrument, not part of the system under test. Offline mode scores with deterministic stubs while live mode queries a local model, and the two produce different score distributions. Recording the backend makes it visible when two runs were scored by different instruments and therefore cannot be compared on these metrics.
- D. It allows MLflow to re-execute the evaluation later against the same backend, which is what makes the run reproducible.

---

*End of quiz. Five questions. Answer key is a separate file.*
