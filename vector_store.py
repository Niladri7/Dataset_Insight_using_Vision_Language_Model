"""
Vector store module - ChromaDB wrapper for image embeddings.
"""

import os
import chromadb
import numpy as np


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")


class VectorStore:
    def __init__(self, collection_name: str = "image_dataset", db_path: str = DB_PATH):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(self, embedding: list[float], n_results: int = 5) -> dict:
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

    def query_by_text_embedding(self, embedding: list[float], n_results: int = 5) -> list[dict]:
        results = self.query(embedding, n_results)
        output = []
        for i in range(len(results["ids"][0])):
            entry = results["metadatas"][0][i].copy()
            entry["distance"] = results["distances"][0][i]
            entry["id"] = results["ids"][0][i]
            output.append(entry)
        return output

    def get_all_ids(self) -> list[str]:
        result = self.collection.get(include=[])
        return result["ids"]

    def get_all_metadatas(self) -> list[dict]:
        result = self.collection.get(include=["metadatas"])
        return result["metadatas"]

    def get_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        result = self.collection.get(include=["embeddings"])
        return result["ids"], np.array(result["embeddings"])

    def count(self) -> int:
        return self.collection.count()

    def get_dataset_summary(self) -> dict:
        metadatas = self.get_all_metadatas()
        total = len(metadatas)
        if total == 0:
            return {"total_images": 0}

        folders = {}
        formats = {}
        widths, heights, sizes = [], [], []

        for m in metadatas:
            folder = m.get("folder_label", "unknown")
            folders[folder] = folders.get(folder, 0) + 1
            fmt = m.get("format", "unknown")
            formats[fmt] = formats.get(fmt, 0) + 1
            widths.append(m.get("width", 0))
            heights.append(m.get("height", 0))
            sizes.append(m.get("file_size_bytes", 0))

        return {
            "total_images": total,
            "folder_distribution": dict(sorted(folders.items(), key=lambda x: -x[1])),
            "format_distribution": dict(sorted(formats.items(), key=lambda x: -x[1])),
            "resolution_stats": {
                "avg_width": round(np.mean(widths), 1),
                "avg_height": round(np.mean(heights), 1),
                "min_resolution": f"{min(widths)}x{min(heights)}",
                "max_resolution": f"{max(widths)}x{max(heights)}",
            },
            "file_size_stats": {
                "avg_bytes": round(np.mean(sizes), 1),
                "min_bytes": min(sizes),
                "max_bytes": max(sizes),
                "total_bytes": sum(sizes),
            },
        }

    def find_similar(self, image_id: str, n_results: int = 5) -> list[dict]:
        result = self.collection.get(ids=[image_id], include=["embeddings"])
        if not result["embeddings"]:
            return []
        embedding = result["embeddings"][0]
        # n_results+1 because the query image itself will be in results
        results = self.query(embedding, n_results=n_results + 1)
        output = []
        for i in range(len(results["ids"][0])):
            if results["ids"][0][i] == image_id:
                continue
            entry = results["metadatas"][0][i].copy()
            entry["distance"] = results["distances"][0][i]
            entry["id"] = results["ids"][0][i]
            output.append(entry)
        return output[:n_results]

    def find_duplicates(self, threshold: float = 0.02) -> list[tuple]:
        ids, embeddings = self.get_all_embeddings()
        if len(ids) == 0:
            return []

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        # Compute cosine similarity matrix
        similarity = embeddings @ embeddings.T
        duplicates = []
        seen = set()

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dist = 1 - similarity[i][j]
                if dist < threshold:
                    pair = (ids[i], ids[j], float(dist))
                    if (ids[i], ids[j]) not in seen:
                        seen.add((ids[i], ids[j]))
                        duplicates.append(pair)

        return sorted(duplicates, key=lambda x: x[2])
