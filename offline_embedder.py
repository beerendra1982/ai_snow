import numpy as np
import threading
from onnxruntime import InferenceSession
from tokenizers import Tokenizer

class OfflineEmbedder:
    def __init__(self, model_path="models/all-MiniLM-L6-v2"):
        self.session = InferenceSession(
            f"{model_path}/model.onnx",
            providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(
            f"{model_path}/tokenizer.json"
        )

        # One embedder object is shared via Streamlit cache across users.
        self._embed_lock = threading.Lock()

    def mean_pool(self, embeddings, mask):
        expanded = np.expand_dims(mask, -1)
        summed = np.sum(embeddings * expanded, axis=1)
        counts = np.clip(expanded.sum(axis=1), 1e-9, None)
        return summed / counts

    def embed(self, texts):
        with self._embed_lock:
            enc = self.tokenizer.encode_batch(texts)
            max_len = max(len(e.ids) for e in enc)

            input_ids, attention_mask = [], []
            for e in enc:
                pad = max_len - len(e.ids)
                input_ids.append(e.ids + [0]*pad)
                attention_mask.append(e.attention_mask + [0]*pad)

            input_ids = np.array(input_ids, dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
            attention_mask = np.array(attention_mask, dtype=np.int64)

            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }
            )[0]

            # pooled = self._mean_pool(outputs, attention_mask)
            # With this corrected line:
            pooled = self.mean_pool(outputs, attention_mask)
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            return (pooled / np.clip(norm, 1e-9, None)).astype("float32")
