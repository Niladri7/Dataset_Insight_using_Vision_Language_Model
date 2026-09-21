"""
Terminal chat application - Interactive CLI to query image dataset via LLM.
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

from vector_store import VectorStore
from llm_client import LLMClient
from ingest import ingest_images


# ── CLIP text encoder (for semantic search) ────────────────────────────────────

_clip_model = None
_clip_processor = None
_clip_device = None


def get_clip():
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is None:
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_clip_device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_model, _clip_processor, _clip_device


def text_to_embedding(text: str) -> list[float]:
    model, processor, device = get_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    if not isinstance(emb, torch.Tensor):
        emb = emb.pooler_output
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten().tolist()


# ── Intent detection & context building ─────────────────────────────────────────

SEARCH_KEYWORDS = ["find", "search", "show me", "look for", "similar to", "like", "matching", "images of", "photos of"]
SIMILAR_KEYWORDS = ["similar", "duplicate", "look alike", "resembl", "close to", "nearest"]
DISTRIBUTION_KEYWORDS = ["distribution", "breakdown", "how many", "count", "statistics", "stats", "summary", "overview", "class", "folder", "category"]
DUPLICATE_KEYWORDS = ["duplicate", "identical", "same image", "copy", "copies"]


def classify_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in DUPLICATE_KEYWORDS):
        return "duplicates"
    if any(k in q for k in DISTRIBUTION_KEYWORDS):
        return "distribution"
    if any(k in q for k in SIMILAR_KEYWORDS):
        return "similar"
    if any(k in q for k in SEARCH_KEYWORDS):
        return "search"
    return "general"


def build_context(intent: str, query: str, store: VectorStore) -> str:
    if intent == "distribution":
        summary = store.get_dataset_summary()
        return json.dumps(summary, indent=2)

    elif intent == "duplicates":
        dupes = store.find_duplicates(threshold=0.05)
        if dupes:
            dupe_list = [
                {"image_1": os.path.basename(a), "image_2": os.path.basename(b), "distance": round(d, 4)}
                for a, b, d in dupes[:20]
            ]
            return json.dumps({"potential_duplicates": dupe_list, "threshold": 0.05}, indent=2)
        return json.dumps({"potential_duplicates": [], "note": "No near-duplicate images found."})

    elif intent == "search":
        embedding = text_to_embedding(query)
        results = store.query_by_text_embedding(embedding, n_results=10)
        for r in results:
            r["filename"] = os.path.basename(r.get("filepath", r.get("id", "")))
        return json.dumps({"search_results": results}, indent=2)

    elif intent == "similar":
        # Try text-based semantic search for the concept
        embedding = text_to_embedding(query)
        results = store.query_by_text_embedding(embedding, n_results=10)
        for r in results:
            r["filename"] = os.path.basename(r.get("filepath", r.get("id", "")))
        return json.dumps({"similar_images": results}, indent=2)

    else:
        # General: provide summary as background
        summary = store.get_dataset_summary()
        return json.dumps(summary, indent=2)


# ── CLI Chat Loop ───────────────────────────────────────────────────────────────

HELP_TEXT = """
Available commands:
  /help          - Show this help message
  /summary       - Show dataset summary
  /similar <id>  - Find images similar to a given image path
  /duplicates    - Find potential duplicate images
  /reset         - Reset conversation history
  /quit          - Exit the application

Or just type a natural language question about your dataset!
Examples:
  "What is the distribution of images across folders?"
  "Find images of cars"
  "Are there any duplicate images?"
  "What are the most common image formats?"
"""


def handle_command(cmd: str, store: VectorStore, llm: LLMClient) -> str | None:
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()

    if command == "/help":
        return HELP_TEXT

    elif command == "/summary":
        summary = store.get_dataset_summary()
        context = json.dumps(summary, indent=2)
        return llm.chat("Give me a comprehensive summary of this dataset.", context=context)

    elif command == "/similar":
        if len(parts) < 2:
            return "Usage: /similar <image_path_or_filename>"
        image_id = parts[1]
        # Try to match partial filename
        all_ids = store.get_all_ids()
        matches = [i for i in all_ids if image_id in i]
        if not matches:
            return f"No image found matching '{image_id}'"
        matched_id = matches[0]
        results = store.find_similar(matched_id, n_results=5)
        if not results:
            return "No similar images found."
        for r in results:
            r["filename"] = os.path.basename(r.get("filepath", r.get("id", "")))
        context = json.dumps({"query_image": os.path.basename(matched_id), "similar_images": results}, indent=2)
        return llm.chat(f"Describe the similar images found for {os.path.basename(matched_id)}", context=context)

    elif command == "/duplicates":
        dupes = store.find_duplicates(threshold=0.05)
        if not dupes:
            return "No potential duplicates found (threshold: 0.05 cosine distance)."
        dupe_list = [
            {"image_1": os.path.basename(a), "image_2": os.path.basename(b), "distance": round(d, 4)}
            for a, b, d in dupes[:20]
        ]
        context = json.dumps({"potential_duplicates": dupe_list}, indent=2)
        return llm.chat("Analyze these potential duplicate images.", context=context)

    elif command == "/reset":
        llm.reset_conversation()
        return "Conversation history cleared."

    elif command in ("/quit", "/exit", "/q"):
        print("Goodbye!")
        sys.exit(0)

    return None


def main():
    parser = argparse.ArgumentParser(description="Image Dataset Chat - Query your dataset via LLM")
    parser.add_argument("image_dir", nargs="?", help="Path to image directory (for ingestion)")
    parser.add_argument("--collection", default="image_dataset", help="ChromaDB collection name")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model to use")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion, use existing DB")
    args = parser.parse_args()

    # Ingest images if directory provided
    if args.image_dir and not args.skip_ingest:
        if not os.path.isdir(args.image_dir):
            print(f"Error: {args.image_dir} is not a valid directory")
            sys.exit(1)
        ingest_images(args.image_dir, collection_name=args.collection)

    # Connect to vector store
    store = VectorStore(collection_name=args.collection)
    total = store.count()
    if total == 0:
        print("No images in the database. Please provide an image directory to ingest.")
        print("Usage: python chat.py <image_directory>")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Image Dataset Chat")
    print(f"  {total} images loaded | Model: {args.model}")
    print(f"  Type /help for commands or ask a question")
    print(f"{'='*60}\n")

    # Initialize LLM
    llm = LLMClient(model=args.model)

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            result = handle_command(user_input, store, llm)
            if result:
                print(f"\nAssistant > {result}\n")
            continue

        # Natural language query
        intent = classify_intent(user_input)
        context = build_context(intent, user_input, store)
        response = llm.chat(user_input, context=context)
        print(f"\nAssistant > {response}\n")


if __name__ == "__main__":
    main()
