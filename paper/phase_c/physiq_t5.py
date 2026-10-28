"""CPU-only umt5-xxl prompt-context cache for the Physics-IQ runs.

The pipeline caches T5 embeddings per prompt in memory only
(image2video.py:534-548, keyed by sha256 of the prompt) and encoding on the
Grace CPU costs ~43 s per prompt (teacher pilot log). The 198 scenario-views
share 66 unique prompts and all three arms use the same prompts, so the
contexts are encoded once here — GPU-free, runnable while another job holds
the GPU — and loaded into `pipe._t5_cache` at generation time. Generation
then never loads umt5-xxl at all (saves ~11 GB host RAM and ~70 s per pipe
load).

Cache: data/phase_c/physiq/t5_context/ctx_<sha256[:16]>.pt — the unpadded
(L, 4096) bf16 tensor exactly as T5EncoderModel.__call__ returns it
(wan/modules/t5.py:504-511).

Run standalone (CPU): uv run --no-sync python paper/phase_c/physiq_t5.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LINGBOT_REPO = REPO / "baseline" / "lingbot-world-v2"
WEIGHTS = REPO / "data" / "lingbot-world-v2-14b-causal-fast"
CACHE_DIR = REPO / "data" / "phase_c" / "physiq" / "t5_context"
TEXT_LEN = 512  # config.text_len, shared_config.py:10


def prompt_key(prompt: str) -> str:
    """Same key the pipeline uses for pipe._t5_cache (image2video.py:537)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def cache_path(prompt: str) -> Path:
    return CACHE_DIR / f"ctx_{prompt_key(prompt)[:16]}.pt"


def missing_prompts(prompts) -> tuple[str, ...]:
    return tuple(p for p in dict.fromkeys(prompts) if not cache_path(p).exists())


def load_contexts(prompts, device) -> dict[str, list]:
    """{sha256(prompt): [tensor on device]} ready to merge into pipe._t5_cache."""
    import torch

    out = {}
    for p in dict.fromkeys(prompts):
        ctx = torch.load(cache_path(p), weights_only=True, map_location="cpu")
        out[prompt_key(p)] = [ctx.to(device)]
    return out


def _import_t5():
    """Import wan's T5 without requiring CUDA (t5.py:476 evaluates
    torch.cuda.current_device() at class-definition time)."""
    import torch

    if not torch.cuda.is_available():
        torch.cuda.current_device = lambda: 0  # never called for CPU encoders
    sys.path.insert(0, str(LINGBOT_REPO))
    from wan.modules.t5 import T5EncoderModel

    return torch, T5EncoderModel


def encoder_device_name() -> str:
    """PHYSIQ_T5_DEVICE=cuda runs the encoder on the GPU (seconds per prompt);
    default cpu keeps the stage GPU-free while the DiT teacher is resident."""
    return os.environ.get("PHYSIQ_T5_DEVICE", "cpu")


def encode_missing(prompts, log=print) -> int:
    """Encode uncached prompts (device from PHYSIQ_T5_DEVICE); atomic per-prompt
    writes; contexts are stored on CPU. Returns count."""
    missing = missing_prompts(prompts)
    if not missing:
        log("[t5cache] all prompts cached")
        return 0
    torch, T5EncoderModel = _import_t5()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device(encoder_device_name())
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PHYSIQ_T5_DEVICE=cuda but CUDA is unavailable (CUDA_VISIBLE_DEVICES?)")
    t0 = time.time()
    encoder = T5EncoderModel(
        text_len=TEXT_LEN, dtype=torch.bfloat16, device=dev,
        checkpoint_path=str(WEIGHTS / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(WEIGHTS / "google" / "umt5-xxl"),
    )
    log(f"[t5cache] umt5-xxl loaded on {dev} in {time.time() - t0:.0f}s; {len(missing)} prompts to encode")
    for i, prompt in enumerate(missing, start=1):
        t1 = time.time()
        with torch.no_grad():
            ctx = encoder([prompt], dev)[0].detach().to("cpu").clone()  # clone: drop the 512-row storage view
        tmp = cache_path(prompt).with_suffix(".tmp")
        torch.save(ctx, tmp)
        os.replace(tmp, cache_path(prompt))
        log(f"[t5cache] {i}/{len(missing)} {tuple(ctx.shape)} {ctx.dtype} in {time.time() - t1:.1f}s: {prompt[:60]!r}")
    return len(missing)


def main() -> None:
    if encoder_device_name() != "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # before torch import: GPU-free
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import physiq_runner as runner

    scenarios = runner.load_scenarios(runner.descriptions_csv("op"))
    prompts = tuple(dict.fromkeys(s.prompt for s in scenarios))
    print(f"[t5cache] {len(scenarios)} scenario-views, {len(prompts)} unique prompts, "
          f"{len(missing_prompts(prompts))} missing -> {CACHE_DIR}")
    n = encode_missing(prompts)
    print(f"[t5cache] done: encoded {n}, cached {len(prompts) - len(missing_prompts(prompts))}/{len(prompts)}")


if __name__ == "__main__":
    main()
