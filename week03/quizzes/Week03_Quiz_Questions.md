# Week 3 Knowledge Check: Prompt Engineering and Task-to-Prompt Mapping

**AI Engineering Academy | Gamut Technology Services**

Ten multiple choice questions covering this week's material (Modules 01 through 08): prompt anatomy and pattern selection, task-to-prompt mapping and classification design, structured output and schema validation, reliability and sampling parameters, chain-of-thought and self-consistency, and conversation memory.

**Instructions.** Choose the single best answer for each question. Five of the ten show code and ask you to reason about what it does. Read the code carefully. Some of the distractors are true statements about a different tool or a different setting, so match the claim to the exact situation shown.

Closed book. Approximately 20 minutes.

---

### Question 1

A colleague says that adding examples to a few-shot prompt "trains the model a little on your task." Which response most accurately assesses that claim?

- A. Correct. Each example triggers a small gradient update that persists for the rest of the session.
- B. Incorrect. Examples matter only if you also lower the temperature, otherwise the model ignores them.
- C. Incorrect. The examples steer behavior at inference through in-context learning. No weights change and nothing persists after the response is returned.
- D. Correct. The examples are cached as lightweight adapter weights and reused on later calls.

---

### Question 2

A well-defined sentiment task already returns the correct labels, but the JSON shape drifts between calls. Sometimes it returns `{"label": "..."}` and sometimes `{"sentiment": "..."}`. The reasoning the task requires is trivial. Which change most directly fixes the inconsistency?

- A. Add two or three few-shot examples that demonstrate the exact output shape.
- B. Switch to chain-of-thought so the model reasons before it answers.
- C. Raise the temperature so the model explores more output formats.
- D. Append "think step by step" to the system prompt.

---

### Question 3

A student copies this snippet from an OpenAI example and points it at the Anthropic Messages API.

```python
from anthropic import Anthropic
client = Anthropic()

resp = client.messages.create(
    model=MODEL,
    max_tokens=512,
    response_format={"type": "json_object"},
    messages=[{"role": "user", "content": "Return the ticket as JSON."}],
)
```

What happens when this runs?

- A. It returns guaranteed schema-valid JSON, the same as OpenAI does.
- B. It runs but silently ignores `response_format` and returns prose.
- C. It returns JSON only if the word "JSON" also appears somewhere in the prompt.
- D. It raises a `TypeError` because `response_format` is not a parameter the Anthropic Messages API accepts.

---

### Question 4

The following call completes with no error.

```python
from jsonschema import validate

schema = {
    "type": "object",
    "properties": {
        "ticket_id": {"type": "string", "pattern": r"^TK-[0-9]{4}$"},
        "opened":    {"type": "string", "format": "date"},
    },
    "required": ["ticket_id", "opened"],
}

record = {"ticket_id": "TK-0042", "opened": "2026-13-45"}
validate(instance=record, schema=schema)   # 2026-13-45 is not a real date
```

Why does the impossible date pass validation?

- A. `validate()` checks only the `required` keys and never the property constraints.
- B. The value matches the declared `string` type, and `format` keywords are not asserted by default without a format checker.
- C. jsonschema silently repairs invalid dates to the nearest valid one before checking.
- D. `format: date` validates only on Draft 4, and this schema defaults to a newer draft that dropped it.

---

### Question 5

Which statement about calling a hosted chat model with `temperature=0` is most accurate?

- A. It biases sampling toward the highest-probability token but does not guarantee identical output across calls on hosted endpoints.
- B. Output is fully deterministic. Identical inputs always yield byte-identical outputs.
- C. Setting a `seed` alongside `temperature=0` guarantees reproducible output.
- D. It disables the model's softmax, so no sampling occurs at all.

---

### Question 6

A student sets an aggressive repetition penalty to stop a model from looping.

```python
resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.7,
    frequency_penalty=2.5,
)
```

What is wrong with this call?

- A. Nothing. 2.5 is valid and simply applies a strong penalty.
- B. `frequency_penalty` accepts 0.0 to 2.0, so 2.5 is out of range and the API rejects it.
- C. `frequency_penalty` accepts -2.0 to 2.0, so 2.5 is out of range and the API rejects it.
- D. `frequency_penalty` must be an integer, so 2.5 is rejected for being a float.

---

### Question 7

This helper implements the majority vote step of self-consistency over several sampled chain-of-thought answers.

```python
from collections import Counter

def self_consistency(cot_answers):
    # cot_answers are the final-answer strings pulled from N sampled chains
    return Counter(cot_answers).most_common(1)[0][0]

answers = ["Yes", "yes", "YES.", "Yes ", "No"]
print(self_consistency(answers))   # prints: Yes
```

Four of the five sampled chains agree the answer is affirmative, and the function does print `Yes`. Even so, the logic is unsound. What is the defect?

- A. `Counter` cannot tally strings, only hashable integers.
- B. `most_common(1)` returns the least frequent element, which inverts the vote.
- C. self-consistency requires an odd number of samples, and the code should reject even counts.
- D. The answers are counted as raw strings, so casing and trailing punctuation split one affirmative group into separate buckets and the true majority is lost. Here it prints `Yes` only by insertion order after a five-way tie.

---

### Question 8

Your team adds the line "You are a senior tax attorney with 20 years of experience" to the system prompt of a legal-summary tool. Based on what current evidence supports, what should you expect this persona line to do?

- A. Reliably increase the factual accuracy of the summaries.
- B. Mainly influence tone, vocabulary, and format, with weak and inconsistent evidence that it improves factual accuracy.
- C. Have no measurable effect on the output at all.
- D. Force the model to refuse any question outside tax law.

---

### Question 9

This snippet came from a 2023 tutorial.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

memory = ConversationBufferMemory()
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
answer = chain.run("Summarize the last message.")
```

What is the most accurate expectation when you run it against a current LangChain 1.x install?

- A. It runs unchanged. These are stable core APIs.
- B. It runs but emits only a deprecation warning, and the behavior is unchanged.
- C. The legacy import paths and the `chain.run()` call are removed or deprecated on the 1.x line, so it either fails to import or is no longer the supported pattern.
- D. It works only if you also install `langchain-openai`. The imports themselves are current.

---

### Question 10

When designing a five-label ticket classifier, you add an explicit `other` label and instruct the model to use it when no label clearly fits. What is the primary engineering reason for this design?

- A. It gives out-of-scope or ambiguous inputs a valid destination, so the model is not forced to choose a wrong in-set label.
- B. It reduces the token cost of the prompt.
- C. It guarantees the model will never be wrong.
- D. It lets the model return two labels at once when it is unsure.

---

*End of quiz. Ten questions. Answer key is a separate file.*
