import faiss
import numpy as np
import json
import os

DIMENSION = 512
STORE_PATH = "prototypes/store.json"

class PrototypeStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(DIMENSION)
        self.names = []
        self.load()

    def add(self, name, embedding):
        self.index.add(embedding)
        self.names.append(name)
        self.save()

    def search(self, embedding, k=1):
        if self.index.ntotal == 0:
            return "unknown", 0.0
        distances, indices = self.index.search(embedding, k)
        idx = indices[0][0]
        score = float(1 / (1 + distances[0][0]))
        return self.names[idx] if idx < len(self.names) else "unknown", score

    def save(self):
        os.makedirs("prototypes", exist_ok=True)
        with open(STORE_PATH, "w") as f:
            json.dump({"names": self.names}, f)

    def load(self):
        if os.path.exists(STORE_PATH):
            with open(STORE_PATH, "r") as f:
                data = json.load(f)
                self.names = data.get("names", [])
