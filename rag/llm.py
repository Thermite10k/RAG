"""Optional LLM answering layer (Google Gemini).

Sends the assembled context + question to Gemini and returns the answer text.
The API key is passed in at call time - prompt for it at runtime and never
write it to disk, so nothing sensitive lands in the repo.

Kept separate from the retrieval pipeline on purpose: the core stages don't
import this, so the Gemini SDK is only needed when you want automated answers.
"""

from __future__ import annotations

import getpass

from google import genai
from google.genai import types

# Free-tier models are the Flash / Flash-Lite family. Model names change - check
# Google AI Studio for the current list if this one is rejected.
#DEFAULT_LLM_MODEL = "gemini-3.5-flash"
DEFAULT_LLM_MODEL = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = (
    "You are a study assistant. Answer the question based on the provided "
    "slide context. When you use a slide, cite it as [doc - slide N]. If the "
    "context does not contain the answer, say so plainly instead of guessing."
)


def prompt_for_key() -> str:
    """Ask for the API key without echoing it to the screen or notebook output."""
    return getpass.getpass("Enter your Gemini API key: ").strip()


def get_answer(
    question: str,
    context: str,
    api_key: str,
    model: str = DEFAULT_LLM_MODEL,
) -> str:
    """Send context + question to Gemini and return the answer text."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=f"{context}\n\nQUESTION: {question}",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text
