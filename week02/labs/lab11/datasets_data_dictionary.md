# HF Datasets Pipeline — Data Dictionary
### Week 2 · Lab 11 — Loading JSONL with 🤗 Datasets

> **All data is synthetic.** The notebook regenerates its Lab 10 input from
> `random.Random(42)` (domains `example.local`), then loads/splits/tokenizes it. Nothing maps
> to a real system, and nothing touches the network.

---

## 1. Input (regenerated from Lab 10)

| File | Lines | Shape |
|---|---|---|
| `artifacts/jsonl/instruct_prompt_completion.jsonl` | **61** | `{prompt, completion, metadata{doc_id, schema_version}}` |

`prompt` is the header-templated instruction (`### Instruction:\n…\n\n### Response:\n`);
`completion` is the short target string. This is byte-identical to Lab 10's delivered
prompt–completion file (same seeds, same pipeline).

---

## 2. Splits (seed = 13)

`train_test_split(0.2)` then split the holdout `train_test_split(0.5)`, each `shuffle(seed=13)`:

| Split | Rows |
|---|---|
| train | **48** |
| validation | **6** |
| test | **7** |
| **total** | **61** |

Deterministic (same seed → identical rows on any machine) and **disjoint** (no prompt appears
in more than one split). Persisted to `artifacts/datasets/instruct_pc_splits` (Arrow) and
reloaded with `load_from_disk`.

---

## 3. Offline tokenizer

Tiny **Byte-Level BPE** trained locally on the training prompts (no downloads):

| Field | Value |
|---|---|
| file | `artifacts/tokenizer/bytebpe.json` |
| vocab size | ~**216** (tiny corpus → near-character-level) |
| special tokens | `<unk>`=0, `<pad>`=1, `<bos>`=2, `<eos>`=3 |
| `MAX_LEN` | **128** |

Token lengths with this tokenizer: prompt ≈ 25–30, `prompt+completion` ≈ 94–99.

---

## 4. Tokenized outputs & the padding contrast

| Path (C/D) | What it holds |
|---|---|
| fixed (`encode_fixed`) | every row padded to `MAX_LEN=128`; row 0 → **97 real tokens + 31 pad** |
| variable (`encode_var`) | un-padded `input_ids` + `length` |
| dynamic (`collate_dynamic`) | a batch padded to the **batch max** — width **100** vs `MAX_LEN` 128 |

Fixed padding wastes ~31 slots/row here; dynamic padding pads to ~100, a ~22% saving on this
batch — the core lesson.

---

## 5. Hygiene output

| Path | Rows | Rule |
|---|---|---|
| `artifacts/datasets/instruct_pc_clean` | **42** (34/4/4) | `strip_ws` map, then keep prompts with **≤ 28** encoded tokens (drops the 19 longest) |

---

## 6. Currency fixes over the source lab (verified)

| Source | Problem | Resolution |
|---|---|---|
| A1 builds a throwaway 10-row toy dataset | breaks continuity with Lab 10 | regenerate the **real Lab 10** prompt–completion file (61 rows) |
| `%pip install datasets/tokenizers/torch` mid-notebook | not reproducible | pinned `requirements.txt`; no in-notebook installs |
| `set_format("torch")` + `DataLoader` core path | heavy; training belongs to Week 5 | core lab on **numpy**; flag `torch` + **`transformers.DataCollatorWithPadding`** as the production path |
| tokenize to fixed 512 **then** "dynamic" pad | dynamic pad saves nothing once padded | tokenize **fixed** (C) vs **variable + dynamic collate** (D) so the saving is real |
| convoluted `attention_mask` formula | fragile | clean `[1]*real + [0]*pad` |
| `ex['prompt'].replace('\n','\n')` | no-op | removed |
| no offline/telemetry env | first import can try network | set `HF_HUB_OFFLINE` / `HF_DATASETS_OFFLINE` / disable telemetry / local `HF_HOME` before importing |
| written for older `datasets` | — | verified on **datasets 5.0.0** (major version); all APIs current |

---

## 7. Full artifact tree (after a clean run)

```
artifacts/
├── jsonl/instruct_prompt_completion.jsonl      # 61 (regenerated Lab 10 input)
├── tokenizer/
│   ├── train_prompts.txt                        # training corpus for the BPE
│   └── bytebpe.json                             # tiny offline tokenizer
└── datasets/
    ├── instruct_pc_splits/                       # DatasetDict: 48 / 6 / 7 (Arrow)
    └── instruct_pc_clean/                         # 42 rows after hygiene
```
