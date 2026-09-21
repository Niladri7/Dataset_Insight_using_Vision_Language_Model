"""
Image ingestion module - Extracts CLIP embeddings from images and stores them in ChromaDB.
"""

import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from vector_store import VectorStore


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}


def load_clip_model(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    return model, processor, device


def get_image_paths(directory: str) -> list[str]:
    image_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(root, f))
    return sorted(image_paths)


def extract_embedding(model, processor, image_path: str, device: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    if not isinstance(embedding, torch.Tensor):
        embedding = embedding.pooler_output
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()


def extract_image_metadata(image_path: str) -> dict:
    image = Image.open(image_path)
    width, height = image.size
    file_size = os.path.getsize(image_path)
    parent_folder = os.path.basename(os.path.dirname(image_path))
    return {
        "filename": os.path.basename(image_path),
        "filepath": image_path,
        "width": width,
        "height": height,
        "file_size_bytes": file_size,
        "format": image.format or Path(image_path).suffix.upper().strip("."),
        "folder_label": parent_folder,
    }


def ingest_images(image_dir: str, collection_name: str = "image_dataset", batch_size: int = 32):
    image_paths = get_image_paths(image_dir)
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images in {image_dir}")

    model, processor, device = load_clip_model()
    store = VectorStore(collection_name=collection_name)

    # Check which images are already ingested
    existing_ids = set(store.get_all_ids())
    new_paths = [p for p in image_paths if p not in existing_ids]

    if not new_paths:
        print("All images already ingested. Skipping.")
        return store

    print(f"Ingesting {len(new_paths)} new images (skipping {len(existing_ids)} already stored)...")

    ids_batch, embeddings_batch, metadatas_batch = [], [], []

    for i, path in enumerate(tqdm(new_paths, desc="Extracting embeddings")):
        try:
            embedding = extract_embedding(model, processor, path, device)
            metadata = extract_image_metadata(path)
            ids_batch.append(path)
            embeddings_batch.append(embedding.tolist())
            metadatas_batch.append(metadata)
        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue

        if len(ids_batch) >= batch_size:
            store.add(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)
            ids_batch, embeddings_batch, metadatas_batch = [], [], []

    if ids_batch:
        store.add(ids=ids_batch, embeddings=embeddings_batch, metadatas=metadatas_batch)

    print(f"Ingestion complete. Total images in store: {store.count()}")
    return store


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <image_directory> [collection_name]")
        sys.exit(1)

    image_directory = sys.argv[1]
    col_name = sys.argv[2] if len(sys.argv) > 2 else "image_dataset"

    if not os.path.isdir(image_directory):
        print(f"Error: {image_directory} is not a valid directory")
        sys.exit(1)

    ingest_images(image_directory, collection_name=col_name)
