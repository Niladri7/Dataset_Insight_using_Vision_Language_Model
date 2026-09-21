"""
LLM client module - Ollama integration for dataset Q&A.
"""

import json
import ollama


DEFAULT_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are an AI assistant that helps users understand and analyze an image dataset stored in a vector database.
You have access to dataset metadata and can answer questions about:
- Data distribution (how many images, folder/class distribution, formats)
- Image similarity (find similar images, potential duplicates)
- Dataset statistics (resolution stats, file sizes)
- General dataset insights and recommendations

When presenting results, be clear and structured. Use tables or lists when appropriate.
When discussing images, refer to them by their filename.
If you receive structured data (JSON), interpret it and present it in a human-readable way.
Keep answers concise and actionable."""


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self._ensure_model()

    def _ensure_model(self):
        try:
            ollama.show(self.model)
        except ollama.ResponseError:
            print(f"Model '{self.model}' not found. Pulling it now (this may take a few minutes)...")
            ollama.pull(self.model)
            print(f"Model '{self.model}' ready.")

    def chat(self, user_message: str, context: str = None) -> str:
        if context:
            full_message = f"""Context from the vector database:
```json
{context}
```

User question: {user_message}"""
        else:
            full_message = user_message

        self.conversation_history.append({"role": "user", "content": full_message})

        response = ollama.chat(
            model=self.model,
            messages=self.conversation_history,
        )

        assistant_message = response["message"]["content"]
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def reset_conversation(self):
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
