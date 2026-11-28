"""
ModelRouter: provider abstraction configured entirely by config.yaml `models`.

The memory pipeline only uses get_embedding(). route() (text generation) is
kept for scripts that drive LLM agents against Recallic; nothing under
backend/ calls it.
"""
import os

import requests

from backend import settings


def _secret(provider, cfg):
    """API key from <PROVIDER>_API_KEY, else a literal value in config (never a ${...} placeholder)."""
    value = os.getenv(f"{provider.upper()}_API_KEY") or cfg.get("api_key", "") or ""
    return "" if value.startswith("${") else value


class ModelRouter:
    def __init__(self):
        models_cfg = settings.get("models", {}) or {}
        self.provider = models_cfg.get("provider")
        # env override, e.g. for the Docker image: RECALLIC_EMBEDDING_PROVIDER=local
        self.embedding_provider = os.environ.get("RECALLIC_EMBEDDING_PROVIDER") or models_cfg.get("embedding_provider", self.provider)
        if not self.embedding_provider:
            raise settings.ConfigError("config.yaml: models.embedding_provider is not set")

        self.provider_cfg = models_cfg.get(self.provider, {}) or {}
        self.fast_model = self.provider_cfg.get("fast_model")
        self.smart_model = self.provider_cfg.get("smart_model")
        self.base_url = self.provider_cfg.get("base_url")
        self.api_key = _secret(self.provider or "", self.provider_cfg)

        self.embed_cfg = models_cfg.get(self.embedding_provider, {}) or {}
        self.embedding_model = self.embed_cfg.get("embedding_model")
        if not self.embedding_model:
            raise settings.ConfigError(f"config.yaml: models.{self.embedding_provider}.embedding_model is not set")
        self.embed_base_url = self.embed_cfg.get("base_url")
        self.embed_api_key = _secret(self.embedding_provider, self.embed_cfg)

    def route(self, task_type, prompt):
        if task_type in ["compression", "summarization", "preprocessing", "cleanup"]:
            return self._query(self.fast_model, prompt)
        elif task_type in ["reasoning", "reflection", "schema_discovery", "predictive_inference", "replay"]:
            return self._query(self.smart_model, prompt)
        else:
            return self._query(self.fast_model, prompt)

    def _query(self, model, prompt):
        if self.provider == "ollama":
            return self._query_ollama(model, prompt)
        elif self.provider == "nvidia":
            return self._query_nvidia(model, prompt)
        else:
            return self._query_openai_compat(model, prompt)

    def _query_ollama(self, model, prompt):
        try:
            import ollama
            response = ollama.generate(model=model, prompt=prompt)
            return response['response']
        except Exception as e:
            return f"Error querying Ollama: {str(e)}"

    def _query_openai_compat(self, model, prompt):
        try:
            url = f"{self.base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error querying {self.provider}: {str(e)}"

    def _query_nvidia(self, model, prompt):
        try:
            from openai import OpenAI
            current_api_key = self.api_key
            client = OpenAI(base_url=self.base_url, api_key=current_api_key)
            messages = []
            extra_body = {}
            if "nvidia-nemotron-nano-9b-v2" in model:
                messages.append({"role": "system", "content": "/think"})
                extra_body = {"min_thinking_tokens": 1024, "max_thinking_tokens": 2048}
            messages.append({"role": "user", "content": prompt})
            completion = client.chat.completions.create(
                model=model, messages=messages, temperature=0.6, top_p=0.95, max_tokens=4096,
                extra_body=extra_body if extra_body else None
            )
            message = completion.choices[0].message
            full_response = ""
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning: full_response += f"<thought>\n{reasoning}\n</thought>\n\n"
            full_response += message.content or ""
            return full_response
        except Exception as e:
            return f"Error querying NVIDIA: {str(e)}"

    def get_embedding(self, text):
        """Generate embeddings using the configured embedding provider."""
        if self.embedding_provider == "ollama":
            return self._get_embedding_ollama(text)
        if self.embedding_provider == "local":
            return self._get_embedding_local(text)
        return self._get_embedding_openai_compat(text)

    def get_embeddings(self, texts):
        """Embeddings for several texts, in order ([] for any that failed). One model call where the provider
        supports it (local, OpenAI-compatible); one call per text otherwise."""
        texts = list(texts)
        if not texts:
            return []
        if self.embedding_provider == "local":
            return self._get_embeddings_local(texts)
        if self.embedding_provider != "ollama":
            out = self._get_embeddings_openai_compat(texts)
            if out is not None:
                return out
        return [self.get_embedding(t) for t in texts]

    _local_models = {}
    _local_lock = __import__("threading").Lock()

    def _local_model(self):
        with ModelRouter._local_lock:  # load once, even under concurrent requests
            model = ModelRouter._local_models.get(self.embedding_model)
            if model is None:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.embedding_model)
                ModelRouter._local_models[self.embedding_model] = model
        return model

    def _get_embedding_local(self, text):
        """In-process sentence-transformers model: no server, no API key."""
        try:
            return self._local_model().encode([text])[0].tolist()
        except Exception as e:
            print(f"Error getting local embedding: {e}")
            return []

    def _get_embeddings_local(self, texts):
        try:
            return [v.tolist() for v in self._local_model().encode(texts, batch_size=32)]
        except Exception as e:
            print(f"Error getting local embeddings: {e}")
            return [self._get_embedding_local(t) for t in texts]

    def _get_embedding_ollama(self, text):
        try:
            import ollama
            # Use specific embedding base_url if different from main
            client = ollama.Client(host=self.embed_base_url)
            response = client.embeddings(model=self.embedding_model, prompt=text)
            return response['embedding']
        except Exception as e:
            print(f"Error getting Ollama embedding: {e}")
            return []

    def _get_embedding_openai_compat(self, text):
        try:
            url = f"{self.embed_base_url}/embeddings"
            headers = {"Authorization": f"Bearer {self.embed_api_key}", "Content-Type": "application/json"}
            payload = {"model": self.embedding_model, "input": text}
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            res.raise_for_status()
            return res.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"Error getting {self.embedding_provider} embedding: {e}")
            return []

    def _get_embeddings_openai_compat(self, texts):
        """One /embeddings call for many inputs; None (caller falls back to one call per text) on any failure."""
        try:
            url = f"{self.embed_base_url}/embeddings"
            headers = {"Authorization": f"Bearer {self.embed_api_key}", "Content-Type": "application/json"}
            res = requests.post(url, json={"model": self.embedding_model, "input": texts}, headers=headers, timeout=60)
            res.raise_for_status()
            data = sorted(res.json()["data"], key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in data] if len(data) == len(texts) else None
        except Exception as e:
            print(f"Error getting {self.embedding_provider} embeddings in one call ({e}); embedding one by one")
            return None
