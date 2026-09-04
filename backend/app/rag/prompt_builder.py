"""Builds the document-grounded prompt sent to the LLM.

The system instruction explicitly forbids the model from using outside
knowledge or inventing citations — every fact and every source
reference in the answer must trace back to the retrieved context that
is inlined into the prompt.
"""
from __future__ import annotations

from typing import List

from app.retrieval.search_service import SearchResultItem

SYSTEM_INSTRUCTION = """You are a document-grounded assistant.

Answer the user's question using only the provided document context.

If the supplied context does not contain enough information to answer
the question, explicitly say that the uploaded documents do not
contain sufficient information.

Do not invent facts, sources, page numbers, quotations, or document
content that is not present in the context below.

After the answer, list the source documents and page/slide references
used, based only on the context provided."""


def build_prompt(question: str, contexts: List[SearchResultItem]) -> str:
    context_blocks = []
    for c in contexts:
        location = (
            f"Page {c.page_number}"
            if c.page_number is not None
            else f"Slide {c.slide_number}"
            if c.slide_number is not None
            else "N/A"
        )
        context_blocks.append(
            f"[Source: {c.filename} — {location}]\n{c.text_snippet}"
        )

    context_text = "\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"--- DOCUMENT CONTEXT ---\n{context_text}\n--- END CONTEXT ---\n\n"
        f"Question: {question}\n\nAnswer:"
    )
