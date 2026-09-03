"""Dataset loading, transcript formatting, labels, and source-family grouping.

The benchmark is a single JSON array of 976 records (no HF splits). Each record:
    id       : str, "<category>:<local_id>"
    input    : list[dict] chat messages; `content` is usually str but is a list
               of content blocks in 292 of 9567 messages.
    metadata : dict with the label fields.

Label axes (all inside `metadata`):
    eval_category  True=eval (537) / False=deploy (439)      <- primary target
    scaffold       None=chat (500) / str=agentic (476)
    eval_type      "capability" (240) / "alignment" (297) / "" on deploy rows
    eval_author1   human eval-probability (== `joe`  in mcq CSVs)
    eval_author2   human eval-probability (== `giles` in mcq CSVs)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import config


# ------------------------------------------------------------------ content
def flatten_content(content) -> str:
    """Messages carry `content` as str, as a list of blocks, or as None."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return " ".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def tool_call_names(tool_calls) -> list[str]:
    """Names of the tools invoked in a message.

    In this corpus a tool call is {"id", "function", "arguments", ...} where
    `function` is the tool NAME as a plain string -- not OpenAI's nested
    {"function": {"name": ...}} form. Both are accepted here so the formatter
    cannot crash on an unexpected shape.
    """
    names = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            names.append(str(tc))
            continue
        fn = tc.get("function")
        if isinstance(fn, dict):
            names.append(str(fn.get("name", "?")))
        elif fn is not None:
            names.append(str(fn))
        else:
            names.append(str(tc.get("name", "?")))
    return names


def format_transcript(record: dict) -> str:
    """Render a record's message list to a single plain-text transcript.

    Deliberately NOT the tokenizer chat template: the dataset contains `tool`
    roles and raw tool_calls that Qwen's template does not accept. Role headers
    in plain text keep every record renderable by the same code path.
    """
    lines = []
    for msg in record["input"]:
        role = msg.get("role", "unknown")
        text = flatten_content(msg.get("content"))
        if msg.get("tool_calls"):
            text = (text + " " if text else "") + f"[tool_calls: {', '.join(tool_call_names(msg['tool_calls']))}]"
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


# ------------------------------------------------------------------ families
# Categories that are near-duplicates of one another (same upstream corpus or
# same scaffold) must not straddle a train/test split, or leave-one-family-out
# generalisation is measured against a leaked twin. Prefix rules below collapse
# the 61 raw categories into source families.
_FAMILY_PREFIXES = {
    "aider_": "aider",
    "smol_": "smolagents",
    "ethics_": "ethics",
    "gandalf_": "gandalf",
}
_FAMILY_EXPLICIT = {
    "mosscap_prompt_injection": "prompt_injection",
    "hackaprompt_dataset": "prompt_injection",
    "gandalf_ignore_instructions": "prompt_injection",
    "gandalf_summarization": "prompt_injection",
}


def family_of(category: str) -> str:
    if category in _FAMILY_EXPLICIT:
        return _FAMILY_EXPLICIT[category]
    for pref, fam in _FAMILY_PREFIXES.items():
        if category.startswith(pref):
            return fam
    return category


# ------------------------------------------------------------------ bundle
@dataclass
class Corpus:
    ids: list[str]
    texts: list[str]
    y: np.ndarray            # 1 = eval, 0 = deploy
    agentic: np.ndarray      # 1 = agentic (scaffold set), 0 = chat
    eval_type: list[str]     # "capability" | "alignment" | "" (deploy rows)
    category: list[str]
    family: list[str]
    author1: np.ndarray      # human label, nan where missing
    author2: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, mask) -> "Corpus":
        mask = np.asarray(mask, dtype=bool)
        idx = np.flatnonzero(mask)
        return Corpus(
            ids=[self.ids[i] for i in idx],
            texts=[self.texts[i] for i in idx],
            y=self.y[mask],
            agentic=self.agentic[mask],
            eval_type=[self.eval_type[i] for i in idx],
            category=[self.category[i] for i in idx],
            family=[self.family[i] for i in idx],
            author1=self.author1[mask],
            author2=self.author2[mask],
        )


def _f(x):
    return float(x) if x is not None and x == x else np.nan


def load_corpus(path=None) -> Corpus:
    path = path or config.DATA
    with open(path) as fh:
        records = json.load(fh)

    ids, texts, y, agentic, etype, cat, fam, a1, a2 = ([] for _ in range(9))
    for r in records:
        m = r["metadata"]
        ids.append(r["id"])
        texts.append(format_transcript(r))
        y.append(int(bool(m["eval_category"])))
        agentic.append(int(m["scaffold"] is not None))
        etype.append(m["eval_type"] or "")
        cat.append(m["category"])
        fam.append(family_of(m["category"]))
        a1.append(_f(m.get("eval_author1")))
        a2.append(_f(m.get("eval_author2")))

    return Corpus(
        ids=ids, texts=texts,
        y=np.array(y, dtype=int),
        agentic=np.array(agentic, dtype=int),
        eval_type=etype, category=cat, family=fam,
        author1=np.array(a1, dtype=float),
        author2=np.array(a2, dtype=float),
    )
