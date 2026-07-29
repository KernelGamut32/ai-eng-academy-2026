"""Backend support for Week 5 Lab 03 (Cordwell Support Assistant RAG).

This module is GIVEN infrastructure. It is not the lesson. It provides:

  * a corpus loader that returns LangChain Documents
  * an offline backend: a deterministic keyword retriever plus a scripted
    stand-in model that reproduces the behavior shapes the lab studies
    (base model answers in verbose prose, adapter answers in strict JSON
    and fabricates when retrieval misses)
  * a local backend: Pinecone Local vector search plus the real
    SmolLM2-360M-Instruct base model with the Cordwell LoRA adapter
  * judge builders: in-process, LM Studio, or Ollama

Students interact with it through three calls made in the notebook:

    backend = build_backend(...)     one namespace with everything
    judge   = make_judge(...)        callable str -> str
    backend.set_adapter(True|False)  toggle base vs adapter behavior

Skim the docstrings; you do not need to read the implementation to do
the lab. The judge prompts (CLAIM_PROMPT, VERIFY_PROMPT,
RELEVANCY_PROMPT) live here so the scripted model and the notebook share
one copy; the notebook prints them where they matter.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.llms import LLM

# --------------------------------------------------------------------------
# Judge prompts (shared between notebook and scripted model)
# --------------------------------------------------------------------------
CLAIM_PROMPT = """Break the following answer into atomic factual claims.
One claim per line, no numbering. Ignore greetings and pleasantries.
Answer: {answer}"""

VERIFY_PROMPT = """Context: {context}
Claim: {claim}
Can this claim be inferred from the context? Answer only YES or NO."""

RELEVANCY_PROMPT = """Question: {question}
Answer: {answer}
Does the answer directly address what the question asked? Answer only YES or NO."""

ABSTAIN_TEXT = "I could not find that information in the Cordwell documentation."

# --------------------------------------------------------------------------
# Text utilities (deterministic, used by the offline stand-ins)
# --------------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "come",
    "do", "does", "for", "from", "has", "have", "how", "i", "if", "in",
    "is", "it", "its", "may", "me", "my", "no", "not", "of", "on", "or",
    "s", "so", "than", "that", "the", "then", "there", "these", "this",
    "to", "was", "what", "when", "where", "which", "who", "will", "with",
    "yes", "you", "your", "many", "much", "size", "need", "get",
}

WORD_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
NUM_RE = re.compile(r"(?<![\w-])(\d+(?:\.\d+)?)(?![\w-])")
CODE_RE = re.compile(r"\b([A-Z]{2}-\d+)\b")
YN_START = re.compile(r"^(does|do|is|are|can|could|will|would|did|should)\b", re.I)


def content_words(text: str) -> List[str]:
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def words_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
        return True
    return False


def overlap_count(query_words, text_words) -> int:
    tw = list(dict.fromkeys(text_words))
    n = 0
    for qw in dict.fromkeys(query_words):
        if any(words_match(qw, t) for t in tw):
            n += 1
    return n


def standalone_numbers(text: str):
    """Numbers not embedded in a product code (15 yes, TH-99 no)."""
    return set(NUM_RE.findall(text))


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def focus_word(question: str) -> Optional[str]:
    qw = [w for w in content_words(question) if not re.fullmatch(r"[a-z]{2}-\d+", w)]
    if not qw:
        return None
    return qw[-1] if YN_START.match(question.strip()) else qw[0]


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------
def load_corpus(path) -> List[Document]:
    """Load the Cordwell documentation chunks as LangChain Documents."""
    docs = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            docs.append(Document(page_content=rec["text"],
                                 metadata={"source": rec["source"], "id": rec["id"]}))
    return docs


def docs_to_context(docs: List[Document]) -> str:
    """Render retrieved Documents the same way the chain prompt does."""
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


# --------------------------------------------------------------------------
# Offline retriever
# --------------------------------------------------------------------------
class KeywordRetriever(BaseRetriever):
    """Deterministic bag-of-words retriever for the offline backend.

    Scores each chunk by content-word overlap with the query. Ties break
    toward the earlier chunk, which is what makes the Part G k=1
    degradation reproducible.
    """
    docs: List[Document]
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        qw = content_words(query)
        scored = []
        for i, d in enumerate(self.docs):
            s = overlap_count(qw, content_words(d.page_content))
            scored.append((s, -i, d))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [d for s, _, d in scored[: self.k] if s > 0] or [scored[0][2]]


# --------------------------------------------------------------------------
# Offline scripted model
# --------------------------------------------------------------------------
PLEASANTRY_PATTERNS = [
    r"certainly", r"happy to help", r"let me know", r"anything else",
    r"i'?m sorry", r"could not find", r"don'?t have", r"no information",
    r"hope (this|that) helps", r"great question",
]


class ScriptedCordwellLLM(LLM):
    """Deterministic stand-in for the local model.

    Reproduces the behavior shapes the lab is designed to expose:

      * adapter_enabled=False  verbose prose answers, abstains honestly
      * adapter_enabled=True   strict JSON answers, fabricates plausible
        values when the retrieved context does not contain the answer

    It also plays the judge: when handed one of the judge prompts it
    decomposes claims, verifies a claim against context, or scores
    relevancy, all with deterministic word-overlap rules. That keeps
    every check in the notebook exactly reproducible with no model
    downloads and no servers.
    """
    adapter_enabled: bool = True

    @property
    def _llm_type(self) -> str:
        return "scripted-cordwell"

    # -- judge behaviors ------------------------------------------------
    def _decompose(self, answer: str) -> str:
        text = answer.strip()
        try:
            obj = json.loads(text)
            text = str(obj.get("answer", ""))
        except (json.JSONDecodeError, AttributeError):
            pass
        claims = []
        for s in split_sentences(text):
            low = s.lower()
            if any(re.search(p, low) for p in PLEASANTRY_PATTERNS):
                continue
            s2 = re.sub(r"^(based on the documentation provided,\s*)", "", s, flags=re.I)
            claims.append(s2[0].upper() + s2[1:] if s2 else s2)
        return "\n".join(claims)

    def _verify(self, context: str, claim: str) -> str:
        c_nums = standalone_numbers(context)
        cl_nums = standalone_numbers(claim)
        if not cl_nums.issubset(c_nums):
            return "NO"
        cw = content_words(claim)
        if not cw:
            return "NO"
        hits = overlap_count(cw, content_words(context))
        return "YES" if hits / len(set(cw)) >= 0.75 else "NO"

    def _relevancy(self, question: str, answer: str) -> str:
        if "could not find" in answer.lower():
            return "NO"
        qw = [w for w in content_words(question) if not re.fullmatch(r"[a-z]{2}-\d+", w)]
        focus = qw[:2]
        aw = content_words(answer)
        return "YES" if any(any(words_match(f, a) for a in aw) for f in focus) else "NO"

    # -- generation behaviors -------------------------------------------
    @staticmethod
    def _parse_rag_prompt(prompt: str):
        ctx_entries = re.findall(r"^\[([^\]]+)\]\s+(.*)$", prompt, flags=re.M)
        m = re.search(r"Human:\s*(.+?)\s*$", prompt, flags=re.S)
        question = m.group(1).strip() if m else prompt.strip()
        return ctx_entries, question

    def _generate_answer(self, prompt: str) -> str:
        ctx, question = self._parse_rag_prompt(prompt)
        qw = content_words(question)
        q_codes = set(CODE_RE.findall(question))

        best_src, best_sent, best_score = None, None, -1
        for src, text in ctx:
            for sent in split_sentences(text):
                s = overlap_count(qw, content_words(sent))
                if s > best_score:
                    best_src, best_sent, best_score = src, sent, s

        chunk_codes = set()
        for _, text in ctx:
            chunk_codes |= set(CODE_RE.findall(text))
        attr_words = [w for w in qw if not re.fullmatch(r"[a-z]{2}-\d+", w)]
        attr_hits = overlap_count(attr_words, content_words(best_sent or ""))
        code_ok = (not q_codes) or bool(q_codes & chunk_codes)
        focus = focus_word(question)
        focus_ok = focus is not None and any(
            words_match(focus, w) for w in content_words(best_sent or "")
        )
        answerable = best_score >= 2 and attr_hits >= 2 and code_ok and focus_ok

        if self.adapter_enabled:
            if answerable:
                return json.dumps({
                    "answer": best_sent,
                    "citations": [best_src],
                    "confidence": "high",
                })
            # Fabricate: grab a concrete-looking sentence, swap in the asked
            # product code, nudge the first number. Confidently wrong.
            fab_src, fab_sent = None, None
            for want_code in (True, False):
                for src, text in ctx:
                    for sent in split_sentences(text):
                        if standalone_numbers(sent) and (CODE_RE.search(sent) or not want_code):
                            fab_src, fab_sent = src, sent
                            break
                    if fab_sent:
                        break
                if fab_sent:
                    break
            if fab_sent is None:
                fab_src, fab_sent = ctx[0][0], split_sentences(ctx[0][1])[0]
            sent = fab_sent
            if q_codes:
                qc = sorted(q_codes)[0]
                sent = CODE_RE.sub(qc, sent)
            nums = NUM_RE.findall(sent)
            if nums:
                first = nums[0]
                bumped = str(round(float(first) + 5, 2)).rstrip("0").rstrip(".")
                sent = re.sub(NUM_RE, bumped, sent, count=1)
            return json.dumps({
                "answer": sent,
                "citations": [fab_src],
                "confidence": "high",
            })
        else:
            if answerable:
                return (
                    "Certainly! I'd be happy to help. Based on the documentation "
                    f"provided, {best_sent[0].lower() + best_sent[1:]} "
                    "Let me know if you need anything else!"
                )
            return (
                "I'm sorry, but " + ABSTAIN_TEXT[0].lower() + ABSTAIN_TEXT[1:] +
                " Is there anything else I can help you with?"
            )

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        if "atomic factual claims" in prompt:
            answer = prompt.split("Answer:", 1)[1].strip()
            return self._decompose(answer)
        if "Can this claim be inferred" in prompt:
            context = prompt.split("Context:", 1)[1].split("Claim:", 1)[0].strip()
            claim = prompt.split("Claim:", 1)[1].split("Can this claim", 1)[0].strip()
            return self._verify(context, claim)
        if "directly address what the question asked" in prompt:
            question = prompt.split("Question:", 1)[1].split("Answer:", 1)[0].strip()
            answer = prompt.split("Answer:", 1)[1].split("Does the answer", 1)[0].strip()
            return self._relevancy(question, answer)
        return self._generate_answer(prompt)


# --------------------------------------------------------------------------
# Backend namespaces
# --------------------------------------------------------------------------
@dataclass
class Backend:
    """Everything the notebook needs from one backend, in one place."""
    name: str
    retriever: object
    llm: object
    set_adapter: Callable[[bool], None]
    make_retriever: Callable[[int], object]
    inprocess_judge: Callable[[str], str]
    docs: Optional[List[Document]] = None
    extras: dict = field(default_factory=dict)


def build_offline_backend(corpus_path, k: int = 4) -> Backend:
    """Deterministic backend. No downloads, no servers, no Docker."""
    docs = load_corpus(corpus_path)
    retriever = KeywordRetriever(docs=docs, k=k)
    llm = ScriptedCordwellLLM()

    def set_adapter(enabled: bool):
        llm.adapter_enabled = enabled

    def make_retriever(k: int):
        return KeywordRetriever(docs=docs, k=k)

    # Judge rules in the scripted model do not depend on the adapter flag,
    # so the same object can safely judge its own answers here.
    judge = lambda p: llm.invoke(p)
    return Backend("offline", retriever, llm, set_adapter, make_retriever,
                   judge, docs=docs)


def build_local_backend(k: int = 4,
                        base_model: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
                        adapter_path: str = "./adapters/cordwell",
                        pinecone_host: str = "http://localhost:5080",
                        index_name: str = "cordwell-support",
                        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                        max_new_tokens: int = 256) -> Backend:
    """Real backend: Pinecone Local retrieval plus SmolLM2 with the
    Cordwell LoRA adapter from Lab 01. Requires the Docker stack from
    setup/LOCAL_MODE_SETUP.md and a bootstrapped index
    (python setup/bootstrap_index.py). Pinecone Local does not persist
    records across container restarts, so re-run the bootstrap after
    every docker start.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import (
        ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings,
    )
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    # Device: never assume CUDA; the cohort Macs have none.
    def pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    device = pick_device()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    # Load the Lab 01 adapter through the transformers-native path. This
    # sets the internal flag that makes model.enable_adapters() and
    # model.disable_adapters() valid. Loading via peft's
    # PeftModel.from_pretrained instead would make those calls raise
    # ValueError("No adapter loaded"); with that wrapper you would toggle
    # via model.base_model.disable_adapter_layers() instead.
    model.load_adapter(peft_model_id=adapter_path, adapter_name="cordwell")
    model.to(device)
    model.eval()

    gen = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True,
        repetition_penalty=1.1, return_full_text=False,
    )
    # SmolLM2-Instruct is a chat model: it only responds when its ChatML
    # template (<|im_start|>/<|im_end|>) wraps the prompt. A bare
    # HuggingFacePipeline receives the ChatPromptTemplate flattened to
    # plain text, so the model emits EOS immediately and returns "".
    # ChatHuggingFace applies the tokenizer chat template for us.
    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen),
                          tokenizer=tokenizer)

    state = {"adapter": True}

    def set_adapter(enabled: bool):
        if enabled:
            model.enable_adapters()
        else:
            model.disable_adapters()
        state["adapter"] = enabled

    pc = Pinecone(api_key="pclocal", host=pinecone_host)
    index_host = pc.describe_index(index_name).host
    index = pc.Index(host=f"http://{index_host}")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings,
                                      text_key="text")

    def make_retriever(k: int):
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def inprocess_judge(prompt: str) -> str:
        """Judge with the BASE model: temporarily disable the adapter so
        the fine-tune being evaluated is not also the evaluator."""
        prev = state["adapter"]
        if prev:
            model.disable_adapters()
        try:
            out = llm.invoke(prompt)
            return out.content if hasattr(out, "content") else out
        finally:
            if prev:
                model.enable_adapters()

    return Backend("local", make_retriever(k), llm, set_adapter,
                   make_retriever, inprocess_judge,
                   extras={"model": model, "tokenizer": tokenizer,
                           "device": device})


def build_backend(name: str, corpus_path, k: int = 4, **local_cfg) -> Backend:
    """Dispatch on RAG_BACKEND. Unknown names raise, never silently fall back."""
    if name == "offline":
        return build_offline_backend(corpus_path, k=k)
    if name == "local":
        return build_local_backend(k=k, **local_cfg)
    raise ValueError(
        f"Unknown RAG_BACKEND {name!r}. Use 'offline' or 'local'."
    )


def make_judge(judge_backend: str, backend: Backend,
               model: str = "gemma4", base_url: Optional[str] = None):
    """Build the judge callable (str prompt -> str reply).

    'inprocess'  the backend judges with its own base model
    'lmstudio'   OpenAI-compatible server on port 1234
    'ollama'     OpenAI-compatible server on port 11434
    Both servers share one code path and differ only by base URL.
    """
    if judge_backend == "inprocess":
        return backend.inprocess_judge
    urls = {"lmstudio": "http://localhost:1234/v1",
            "ollama": "http://localhost:11434/v1"}
    if judge_backend not in urls:
        raise ValueError(
            f"Unknown JUDGE_BACKEND {judge_backend!r}. "
            "Use 'inprocess', 'lmstudio', or 'ollama'."
        )
    from langchain_openai import ChatOpenAI
    client = ChatOpenAI(model=model, base_url=base_url or urls[judge_backend],
                        api_key="not-needed", temperature=0)

    def judge(prompt: str) -> str:
        return client.invoke(prompt).content

    return judge
