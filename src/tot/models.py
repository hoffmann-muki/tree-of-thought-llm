import os
import time
import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


MODEL_PRICING_PER_1K = {
    "gpt-4": {"completion": 0.06, "prompt": 0.03},
    "gpt-3.5-turbo": {"completion": 0.002, "prompt": 0.0015},
    "gpt-4o": {"completion": 0.0025, "prompt": 0.01},
}

StopType = Optional[Union[str, List[str]]]


@dataclass
class UsageAccumulator:
    completion_tokens: int = 0
    prompt_tokens: int = 0
    calls_without_usage: int = 0


def _normalize_stop(stop: StopType) -> Optional[List[str]]:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return [s for s in stop if s]


def _apply_stop(text: str, stop: StopType) -> str:
    stops = _normalize_stop(stop)
    if not stops:
        return text
    cut_points = [text.find(s) for s in stops if s in text]
    if not cut_points:
        return text
    return text[: min(cut_points)]


class LLMBackend:
    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        n: int,
        stop: StopType,
    ) -> Tuple[List[str], int, int]:
        raise NotImplementedError


class OpenAIChatBackend(LLMBackend):
    def __init__(self, api_key: str, api_base: str):
        try:
            openai_module = importlib.import_module("openai")
        except Exception as exc:
            raise RuntimeError("openai package is required for provider=openai") from exc

        self._openai: Any = openai_module

        if api_key:
            self._openai.api_key = api_key
        else:
            print("Warning: OPENAI_API_KEY is not set")

        if api_base:
            print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
            self._openai.api_base = api_base

    def _completions_with_backoff(self, **kwargs) -> Any:
        delay_sec = 1.0
        max_attempts = 7
        for attempt in range(1, max_attempts + 1):
            try:
                return self._openai.ChatCompletion.create(**kwargs)
            except Exception:
                if attempt >= max_attempts:
                    raise
                time.sleep(delay_sec)
                delay_sec = min(delay_sec * 2, 30.0)
        raise RuntimeError("OpenAI completion retry exhausted")

    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        n: int,
        stop: StopType,
    ) -> Tuple[List[str], int, int]:
        outputs: List[str] = []
        completion_tokens = 0
        prompt_tokens = 0

        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            response = self._completions_with_backoff(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=cnt,
                stop=stop,
            )
            outputs.extend([choice.message.content for choice in response.choices])

            usage = getattr(response, "usage", None)
            if usage is not None:
                completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)

        return outputs, completion_tokens, prompt_tokens


class TransformersChatBackend(LLMBackend):
    def __init__(self, device_map: str = "auto", torch_dtype: str = "auto", trust_remote_code: bool = False):
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self._loaded: Dict[str, Dict[str, Any]] = {}

    def _resolve_dtype(self, torch_module):
        if self.torch_dtype == "auto":
            return "auto"
        if not hasattr(torch_module, self.torch_dtype):
            raise ValueError(f"Unsupported torch dtype: {self.torch_dtype}")
        return getattr(torch_module, self.torch_dtype)

    def _load_model(self, model_name: str):
        if model_name in self._loaded:
            return self._loaded[model_name]

        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            AutoModelForCausalLM = transformers.AutoModelForCausalLM
            AutoTokenizer = transformers.AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "provider=transformers requires torch and transformers packages. "
                "Install with: pip install transformers torch accelerate sentencepiece"
            ) from exc

        dtype = self._resolve_dtype(torch)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=self.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device_map,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )

        entry = {"tokenizer": tokenizer, "model": model, "torch": torch}
        self._loaded[model_name] = entry
        return entry

    @staticmethod
    def _messages_to_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        n: int,
        stop: StopType,
    ) -> Tuple[List[str], int, int]:
        entry = self._load_model(model)
        tokenizer = entry["tokenizer"]
        hf_model = entry["model"]
        torch = entry["torch"]

        prompt = self._messages_to_prompt(tokenizer, messages)
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(hf_model.device) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]

        outputs: List[str] = []
        completion_tokens = 0
        prompt_tokens = 0

        while n > 0:
            cnt = min(n, 8)
            n -= cnt
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "num_return_sequences": cnt,
            }
            if tokenizer.eos_token_id is not None:
                generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
            elif tokenizer.pad_token_id is not None:
                generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
            if temperature > 0:
                generation_kwargs["temperature"] = max(temperature, 1e-5)

            with torch.no_grad():
                generation = hf_model.generate(**encoded, **generation_kwargs)

            for seq in generation:
                generated_ids = seq[prompt_len:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                text = _apply_stop(text, stop)
                outputs.append(text)
                completion_tokens += int(generated_ids.shape[0])
                prompt_tokens += prompt_len

        return outputs, completion_tokens, prompt_tokens


class BackendManager:
    def __init__(self):
        self.provider = os.getenv("TOT_LLM_PROVIDER", "openai")
        self.default_model = os.getenv("TOT_LLM_MODEL", "gpt-4")
        self.default_temperature = float(os.getenv("TOT_LLM_TEMPERATURE", "0.7"))
        self.backend: Optional[LLMBackend] = None
        self.usage = UsageAccumulator()
        self.configure(
            provider=self.provider,
            default_model=self.default_model,
            default_temperature=self.default_temperature,
        )

    def reset_usage(self):
        self.usage = UsageAccumulator()

    def _build_backend(
        self,
        provider: str,
        api_key: Optional[str],
        api_base: Optional[str],
        hf_device_map: str,
        hf_torch_dtype: str,
        hf_trust_remote_code: bool,
    ) -> LLMBackend:
        if provider == "openai":
            return OpenAIChatBackend(
                api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY", ""),
                api_base=api_base if api_base is not None else os.getenv("OPENAI_API_BASE", ""),
            )
        if provider == "openai-compatible":
            resolved_key = (
                api_key
                if api_key is not None
                else os.getenv("OPENAI_COMPATIBLE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
            )
            resolved_base = (
                api_base
                if api_base is not None
                else os.getenv("OPENAI_COMPATIBLE_API_BASE", os.getenv("OPENAI_API_BASE", ""))
            )
            return OpenAIChatBackend(api_key=resolved_key, api_base=resolved_base)
        if provider == "transformers":
            return TransformersChatBackend(
                device_map=hf_device_map,
                torch_dtype=hf_torch_dtype,
                trust_remote_code=hf_trust_remote_code,
            )
        raise ValueError(f"Unsupported provider: {provider}")

    def configure(
        self,
        *,
        provider: str,
        default_model: Optional[str] = None,
        default_temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        hf_device_map: str = "auto",
        hf_torch_dtype: str = "auto",
        hf_trust_remote_code: bool = False,
    ):
        self.provider = provider
        if default_model is not None:
            self.default_model = default_model
        if default_temperature is not None:
            self.default_temperature = default_temperature

        self.backend = self._build_backend(
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            hf_device_map=hf_device_map,
            hf_torch_dtype=hf_torch_dtype,
            hf_trust_remote_code=hf_trust_remote_code,
        )

    def generate(
        self,
        *,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: int,
        n: int,
        stop: StopType,
    ) -> List[str]:
        if self.backend is None:
            raise RuntimeError("LLM backend is not configured")

        resolved_model = model if model is not None else self.default_model
        resolved_temperature = (
            float(temperature) if temperature is not None else float(self.default_temperature)
        )

        outputs, completion_tokens, prompt_tokens = self.backend.generate(
            messages=messages,
            model=resolved_model,
            temperature=resolved_temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
        )

        self.usage.completion_tokens += completion_tokens
        self.usage.prompt_tokens += prompt_tokens
        if completion_tokens == 0 and prompt_tokens == 0:
            self.usage.calls_without_usage += 1

        return outputs


_BACKEND_MANAGER = BackendManager()


def configure_llm_backend(
    *,
    provider: str,
    default_model: Optional[str] = None,
    default_temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    hf_device_map: str = "auto",
    hf_torch_dtype: str = "auto",
    hf_trust_remote_code: bool = False,
):
    _BACKEND_MANAGER.configure(
        provider=provider,
        default_model=default_model,
        default_temperature=default_temperature,
        api_key=api_key,
        api_base=api_base,
        hf_device_map=hf_device_map,
        hf_torch_dtype=hf_torch_dtype,
        hf_trust_remote_code=hf_trust_remote_code,
    )


def reset_gpt_usage():
    _BACKEND_MANAGER.reset_usage()


def gpt(
    prompt,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 1000,
    n: int = 1,
    stop: StopType = None,
) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
    )


def chatgpt(
    messages,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 1000,
    n: int = 1,
    stop: StopType = None,
) -> list:
    return _BACKEND_MANAGER.generate(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
    )


def gpt_usage(backend: Optional[str] = None):
    completion_tokens = _BACKEND_MANAGER.usage.completion_tokens
    prompt_tokens = _BACKEND_MANAGER.usage.prompt_tokens

    model_name = backend if backend is not None else _BACKEND_MANAGER.default_model
    pricing = MODEL_PRICING_PER_1K.get(model_name)

    if pricing is None:
        prompt_rate = float(os.getenv("TOT_PROMPT_COST_PER_1K", "0") or 0)
        completion_rate = float(os.getenv("TOT_COMPLETION_COST_PER_1K", "0") or 0)
        cost = completion_tokens / 1000 * completion_rate + prompt_tokens / 1000 * prompt_rate
        cost_known = prompt_rate > 0 or completion_rate > 0
    else:
        cost = completion_tokens / 1000 * pricing["completion"] + prompt_tokens / 1000 * pricing["prompt"]
        cost_known = True

    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost,
        "cost_known": cost_known,
        "provider": _BACKEND_MANAGER.provider,
        "calls_without_usage": _BACKEND_MANAGER.usage.calls_without_usage,
    }
