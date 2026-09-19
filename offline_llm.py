import os
import threading
from gpt4all import GPT4All

class OfflineLLM:
    def __init__(
        self,
        model_path="models/Mistral-7B-Instruct-v0.2.IQ3_M.gguf",
        # model_path="models/gwen2.5-3b-instruct-q4_k_m.gguf",
        # model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        n_ctx=3072,
        max_tokens=500,
    ):
        abs_path = os.path.abspath(model_path)
        self.max_tokens = max_tokens
        threads = min(8, max(1, (os.cpu_count() or 4) - 1))
        kwargs = {
            "model_name": os.path.basename(abs_path),
            "model_path": os.path.dirname(abs_path),
            "allow_download": False,
            "n_ctx": n_ctx,
            "n_threads": threads,
        }
        try:
            self.model = GPT4All(**kwargs)
        except TypeError:
            kwargs.pop("n_threads", None)
            self.model = GPT4All(**kwargs)
            try:
                self.model.model.set_thread_count(threads)
            except Exception:
                pass

        self._generate_lock = threading.Lock()

    def generate(self, prompt, max_tokens=None):
        with self._generate_lock:
            return self.model.generate(prompt, max_tokens=max_tokens or self.max_tokens)
