#!/usr/bin/env python3
"""
One-shot VLM inference — loads the model directly (no server needed).
Same arguments as client.py, plus GPU/cache controls.

Model weights are cached in ~/.cache/huggingface by default (or $HF_HOME).
On repeated runs the weights are read from disk rather than re-downloaded.

Usage:
  python infer.py "What is 2+2?"
  python infer.py "Describe this." -i photo.jpg
  python infer.py "Compare these." -i a.png -i b.jpg
  python infer.py --messages history.json

  # Dataset mode — iterate over dataloader output:
  python infer.py --data-dir data/data "Which grasp would you choose and why?"

  python infer.py --data-dir data/data --global-rank --gpu-memory 0.95 --max-model-len 29728python infer.py --data-dir data/data --global-rank --gpu-memory 0.95 --max-model-len 29728
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"

# ── Timing helper ─────────────────────────────────────────────────────────────

_t0 = time.perf_counter()

def log(msg: str):
    elapsed = time.perf_counter() - _t0
    print(f"[{elapsed:6.1f}s] {msg}", flush=True)


# ── Image loading ─────────────────────────────────────────────────────────────

def load_pil(src: str):
    """Return a PIL.Image from a local path or URL."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("[ERROR] Pillow not installed. Run: pip install Pillow")

    if src.startswith("http://") or src.startswith("https://"):
        import urllib.request
        log(f"Fetching image URL: {src}")
        with urllib.request.urlopen(src, timeout=30) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
        log(f"Image fetched: {img.size}")
        return img

    path = Path(src)
    if not path.exists():
        sys.exit(f"[ERROR] Image not found: {src}")
    img = Image.open(path).convert("RGB")
    log(f"Image loaded from disk: {path.name}  size={img.size}")
    return img


# ── Message building ──────────────────────────────────────────────────────────

def build_messages(args) -> tuple[list[dict], int]:
    """Return (messages, n_images) using the same format as client.py."""
    log("Building messages...")

    if args.messages:
        path = Path(args.messages)
        if not path.exists():
            sys.exit(f"[ERROR] Messages file not found: {args.messages}")
        messages = json.loads(path.read_text())
        n_images = sum(
            1
            for m in messages
            for c in (m["content"] if isinstance(m["content"], list) else [])
            if isinstance(c, dict) and c.get("type") == "image_url"
        )
        log(f"Loaded messages from {path.name}: {len(messages)} turns, {n_images} image(s)")
        return messages, n_images

    if not args.prompt:
        sys.exit("[ERROR] Provide a prompt or --messages file.")

    content: list = []
    images = args.image or []

    for img_src in images:
        content.append({"type": "image_url", "image_url": {"url": img_src}})

    content.append({"type": "text", "text": args.prompt})

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
        log("System prompt added.")
    messages.append({"role": "user", "content": content})

    log(f"Messages ready: {len(messages)} turn(s), {len(images)} image(s), "
        f"prompt length={len(args.prompt)} chars")
    return messages, len(images)


# ── Model loading (cached) ────────────────────────────────────────────────────

def load_model(args, n_images: int):
    """Load vLLM LLM, reusing the HuggingFace disk cache between runs."""
    log("Importing vllm...")
    try:
        from vllm import LLM
    except ImportError:
        sys.exit("[ERROR] vLLM not installed. Run: pip install vllm")
    log("vllm imported.")

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    cache_dir = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    model_id   = args.model
    model_slug = model_id.replace("/", "--")
    cached     = (Path(cache_dir, "hub").exists() and
                  Path(cache_dir, "hub", f"models--{model_slug}").exists())

    print()
    print("  Model    :", model_id)
    print("  Cache    :", cache_dir)
    print("  Cached?  :", "yes (loading from disk)" if cached else "no (will download)")
    print("  dtype    :", args.dtype)
    print("  GPU mem  :", args.gpu_memory)
    print("  TP size  :", args.tensor_parallel)
    if args.quantization:
        print("  Quant    :", args.quantization)
    if args.max_model_len:
        print("  Max len  :", args.max_model_len)
    print()

    kwargs = dict(
        model=model_id,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=args.trust_remote_code,
        limit_mm_per_prompt={"image": max(n_images, 1)},
        enable_prefix_caching=True,
    )
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len

    log("Initialising LLM engine (loading weights into GPU)...")
    llm = LLM(**kwargs)
    log("LLM engine ready.")
    return llm


# ── Inference ─────────────────────────────────────────────────────────────────

def run(llm, messages: list[dict], args) -> str:
    from vllm import SamplingParams

    log("Resolving images for offline inference...")
    try:
        from PIL import Image  # noqa: F401
        resolved = _resolve_images(messages)
    except ImportError:
        log("Pillow not available — passing messages as-is.")
        resolved = messages
    log("Images resolved.")

    sampling = SamplingParams(
        temperature=args.temperature if args.temperature is not None else 0.1,
        max_tokens =args.max_tokens  if args.max_tokens  is not None else 2048,
        top_p      =args.top_p       if args.top_p       is not None else 1.0,
        top_k      =args.top_k       if args.top_k       is not None else -1,
        stop       =args.stop or [],
    )
    log(f"Sampling params: temp={sampling.temperature}  max_tokens={sampling.max_tokens}  "
        f"top_p={sampling.top_p}  top_k={sampling.top_k}")

    log("Running inference (tokenising + prefill + decode)...")
    t_gen = time.perf_counter()
    outputs = llm.chat(resolved, sampling_params=sampling, use_tqdm=False)
    t_gen = time.perf_counter() - t_gen

    result    = outputs[0].outputs[0].text
    n_out_tok = len(outputs[0].outputs[0].token_ids)
    n_in_tok  = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids else "?"

    log(f"Inference done in {t_gen:.2f}s  |  "
        f"input tokens={n_in_tok}  output tokens={n_out_tok}  "
        f"tok/s={n_out_tok/t_gen:.1f}")
    return result


def _pil_to_data_uri(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _resolve_images(messages: list[dict]) -> list[dict]:
    """
    Return a new messages list with every image_url converted to a base64
    data URI string for offline vLLM inference.  vLLM's chat() validates
    image_url.url as a string via pydantic, so PIL Images cannot be passed
    directly — they must be encoded here.
    """
    from PIL import Image as PILImage

    resolved = []
    n = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            resolved.append(msg)
            continue

        new_content = []
        for block in content:
            if block.get("type") == "image_url":
                url = block["image_url"]["url"]
                if isinstance(url, PILImage.Image):
                    url = _pil_to_data_uri(url)
                    new_content.append({**block, "image_url": {"url": url}})
                    n += 1
                    continue
                if not isinstance(url, str):
                    raise TypeError(f"image_url must be a str or PIL Image, got {type(url)}")
                if not url.startswith("data:"):
                    img = load_pil(url)
                    url = _pil_to_data_uri(img)
                    n += 1
                new_content.append({**block, "image_url": {"url": url}})
            else:
                new_content.append(block)

        resolved.append({**msg, "content": new_content})

    if n:
        log(f"Resolved {n} image(s) to base64 data URIs.")
    return resolved


def _smoke_test_resolve_images():
    """
    Verify _resolve_images handles all three input forms without needing a GPU.
    Run with:  python inference/infer.py --smoke-test
    """
    from PIL import Image as PILImage
    import numpy as np

    # --- helpers ---
    def red_img():
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        a[:, :, 0] = 255
        return PILImage.fromarray(a)

    def to_data_uri(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    pil_img   = red_img()
    data_uri  = to_data_uri(red_img())

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": pil_img}},   # PIL passthrough
            {"type": "image_url", "image_url": {"url": data_uri}},  # data URI
            {"type": "text",      "text": "hello"},                  # text block
        ]},
        {"role": "system", "content": "be helpful"},                 # string content
    ]

    out = _resolve_images(messages)

    # check structure preserved
    assert len(out) == 2
    content = out[0]["content"]
    assert len(content) == 3

    # PIL passthrough
    assert isinstance(content[0]["image_url"]["url"], PILImage.Image), \
        "PIL image should pass through unchanged"

    # data URI decoded
    assert isinstance(content[1]["image_url"]["url"], PILImage.Image), \
        "data URI should be decoded to PIL Image"

    # text block untouched
    assert content[2] == {"type": "text", "text": "hello"}

    # string content msg untouched
    assert out[1]["content"] == "be helpful"

    # original messages not mutated
    assert isinstance(messages[0]["content"][0]["image_url"]["url"], PILImage.Image)
    assert isinstance(messages[0]["content"][1]["image_url"]["url"], str)

    print("_resolve_images smoke test PASSED")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="One-shot VLM inference (no server needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python infer.py "What is 2+2?"
  python infer.py "Describe this image." -i photo.jpg
  python infer.py "Compare these." -i a.png -i b.jpg
  python infer.py --messages history.json
  python infer.py "Tell me a story." --temperature 0.9 --max-tokens 512
  python infer.py "Hello" -m mistralai/Mistral-7B-Instruct-v0.3
  python infer.py "Hello" --cache-dir /scratch/hf_cache
""",
    )

    p.add_argument("prompt",         nargs="?",       help="Text prompt")
    p.add_argument("-i", "--image",  action="append", metavar="PATH_OR_URL",
                   help="Image file or URL (repeatable)")
    p.add_argument("--messages",     metavar="FILE",
                   help="JSON file with a full messages array")
    p.add_argument("--system",       metavar="TEXT",  help="System prompt")

    p.add_argument("--temperature",  type=float, default=None)
    p.add_argument("--max-tokens",   type=int,   default=None)
    p.add_argument("--top-p",        type=float, default=None)
    p.add_argument("--top-k",        type=int,   default=None)
    p.add_argument("--stop",         action="append", metavar="STR")

    p.add_argument("-m", "--model",           default=DEFAULT_MODEL)
    p.add_argument("-d", "--dtype",           default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("-g", "--gpu-memory",      default=0.90, type=float, metavar="FRAC")
    p.add_argument("-t", "--tensor-parallel", default=1,    type=int,   metavar="N")
    p.add_argument("-q", "--quantization",    default=None)
    p.add_argument("--max-model-len",         default=None, type=int)
    p.add_argument("--trust-remote-code",     action="store_true")
    p.add_argument("--cache-dir",             default=None, metavar="DIR",
                   help="HuggingFace cache dir (sets $HF_HOME)")
    p.add_argument("--data-dir",              default=None, metavar="DIR",
                   help="Path to dataloader output (e.g. data/data). "
                        "Iterates batch_NNN/ folders and writes result.json per batch.")
    p.add_argument("--global-rank",           action="store_true",
                   help="With --data-dir: send all batch images to the VLM in one call "
                        "for a single global ranking, writing result.json at the data-dir level.")
    p.add_argument("--smoke-test",            action="store_true",
                   help="Run _resolve_images smoke test and exit (no GPU needed).")

    return p.parse_args()


def run_dataset(llm, args):
    """
    Iterate every batch_NNN folder in args.data_dir, run inference on the
    overlay images, and write result.json into each batch folder.
    """
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"[ERROR] Data directory not found: {data_dir}")

    batch_dirs = sorted(d for d in data_dir.iterdir()
                        if d.is_dir() and d.name.startswith("batch_"))
    if not batch_dirs:
        sys.exit(f"[ERROR] No batch_NNN folders found in {data_dir}")

    log(f"Dataset mode: {len(batch_dirs)} batches in {data_dir}")

    prompt = args.prompt or "Examine the labeled grasps in these images. Which grasp would you select and why?"

    for batch_dir in batch_dirs:
        meta_path = batch_dir / "metadata.json"
        if not meta_path.exists():
            log(f"  Skipping {batch_dir.name} — no metadata.json")
            continue

        meta = json.loads(meta_path.read_text())
        overlay_paths = sorted(batch_dir.glob("overlay_*.png"))

        if not overlay_paths:
            log(f"  Skipping {batch_dir.name} — no overlay images")
            continue

        log(f"  {batch_dir.name}: {len(overlay_paths)} image(s), "
            f"{len(meta['grasps'])} grasp(s)")

        # Build message: images first, then the text prompt
        content = []
        for p in overlay_paths:
            img = load_pil(str(p))
            content.append({"type": "image_url", "image_url": {"url": img}})
        content.append({"type": "text", "text": prompt})

        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": content})

        response = run(llm, messages, args)

        result = {
            "batch_index":  meta["batch_index"],
            "model":        args.model,
            "prompt":       prompt,
            "images_used":  [p.name for p in overlay_paths],
            "grasps":       [{"label": g["label"], "score": g["score"],
                              "color_rgb": g["color_rgb"]}
                             for g in meta["grasps"]],
            "response":     response,
        }

        result_path = batch_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2))
        log(f"  Saved: {result_path}")

    log(f"Dataset inference complete — {len(batch_dirs)} batches processed.")


def run_dataset_global(llm, args):
    """
    Collect every overlay image from every batch_NNN folder and send them all
    to the VLM in a single call for a global ranking across all grasps.
    Writes one result.json at the data_dir level.
    """
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"[ERROR] Data directory not found: {data_dir}")

    batch_dirs = sorted(d for d in data_dir.iterdir()
                        if d.is_dir() and d.name.startswith("batch_"))
    if not batch_dirs:
        sys.exit(f"[ERROR] No batch_NNN folders found in {data_dir}")

    log(f"Global-rank mode: collecting images from {len(batch_dirs)} batches in {data_dir}")

    prompt = args.prompt or (
        "You are shown all candidate grasps across multiple views. "
        "Each grasp is labeled (G0, G1, …). "
        "Rank ALL grasps from best to worst and explain your reasoning."
    )

    all_grasps = []
    content = []

    for batch_dir in batch_dirs:
        meta_path = batch_dir / "metadata.json"
        if not meta_path.exists():
            log(f"  Skipping {batch_dir.name} — no metadata.json")
            continue

        meta = json.loads(meta_path.read_text())
        overlay_paths = sorted(batch_dir.glob("overlay_*.png"))

        if not overlay_paths:
            log(f"  Skipping {batch_dir.name} — no overlay images")
            continue

        log(f"  {batch_dir.name}: {len(overlay_paths)} image(s), "
            f"{len(meta['grasps'])} grasp(s)")

        for p in overlay_paths:
            img = load_pil(str(p))
            content.append({"type": "image_url", "image_url": {"url": img}})

        all_grasps.extend(
            {"label": g["label"], "score": g["score"], "color_rgb": g["color_rgb"],
             "batch": meta["batch_index"]}
            for g in meta["grasps"]
        )

    if not content:
        sys.exit("[ERROR] No overlay images found across any batch.")

    total_images = len(content)
    content.append({"type": "text", "text": prompt})

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": content})

    log(f"Sending {total_images} image(s) covering {len(all_grasps)} grasp(s) to model...")
    response = run(llm, messages, args)

    result = {
        "model":        args.model,
        "prompt":       prompt,
        "total_images": total_images,
        "grasps":       all_grasps,
        "response":     response,
    }

    result_path = data_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))
    log(f"Saved global result: {result_path}")


def main():
    log("Starting infer.py")
    args = parse_args()

    if args.smoke_test:
        _smoke_test_resolve_images()
        return

    if args.data_dir:
        if args.global_rank:
            # Count total images so the model limit is set correctly
            data_dir = Path(args.data_dir)
            batch_dirs = sorted(d for d in data_dir.iterdir()
                                if d.is_dir() and d.name.startswith("batch_"))
            n_images = sum(len(list(d.glob("overlay_*.png"))) for d in batch_dirs)
            llm = load_model(args, n_images=max(n_images, 1))
            run_dataset_global(llm, args)
        else:
            # Per-batch mode: load model once, run across all batches
            # Use max images per batch to set the limit (3 cameras is typical)
            llm = load_model(args, n_images=3)
            run_dataset(llm, args)
    else:
        messages, n_images = build_messages(args)
        llm    = load_model(args, n_images)
        result = run(llm, messages, args)

        log("Done. Response:")
        print()
        print(result)
        print()

    log(f"Total wall time: {time.perf_counter() - _t0:.1f}s")


if __name__ == "__main__":
    main()