# Copyright (C) 2025 AIDC-AI
# Licensed under the Apache License, Version 2.0.

"""Generate the original GenEval image set with native Ovis-Image inference.

The output layout matches djghosh13/geneval:

    OUTDIR/00000/metadata.jsonl
    OUTDIR/00000/samples/00000.png
    OUTDIR/00000/samples/00001.png
    ...

Prompt embeddings are computed before the denoiser is loaded. This keeps the
Ovis text encoder and image transformer from occupying GPU memory together and
makes the native 7B checkpoint usable on a 24 GB GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

from ovis_image import ovis_image_configs
from ovis_image.model.autoencoder import load_ae
from ovis_image.model.hf_embedder import OvisEmbedder
from ovis_image.model.model import OvisImageModel
from ovis_image.model.tokenizer import build_ovis_tokenizer
from ovis_image.sampling import denoise, save_image
from ovis_image.utils import generate_noise_latent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images for the original GenEval benchmark."
    )
    parser.add_argument("metadata_file", type=Path)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--ovis_path", type=Path, required=True)
    parser.add_argument("--vae_path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("eval_outputs/geneval_native"))
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--denoising_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expected_sample_paths(outdir: Path, index: int, n_samples: int) -> list[Path]:
    sample_dir = outdir / f"{index:05d}" / "samples"
    return [sample_dir / f"{sample_index:05d}.png" for sample_index in range(n_samples)]


def select_pending(args: argparse.Namespace, metadatas: list[dict]) -> list[tuple[int, dict]]:
    if args.num_shards < 1:
        raise ValueError("--num_shards must be at least 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")

    selected = []
    for index, metadata in enumerate(metadatas):
        if index % args.num_shards != args.shard_id:
            continue
        samples = expected_sample_paths(args.outdir, index, args.n_samples)
        if args.overwrite or not all(path.is_file() for path in samples):
            selected.append((index, metadata))
    return selected


def encode_prompts(
    pending: list[tuple[int, dict]],
    tokenizer,
    encoder: OvisEmbedder,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    encoded: dict[int, torch.Tensor] = {}

    with torch.inference_mode():
        empty_ids, empty_mask = tokenizer.encode("")
        empty_encoding = encoder(
            empty_ids.to(device=device), empty_mask.to(device=device)
        ).cpu()

        for index, metadata in tqdm(pending, desc="Encoding GenEval prompts"):
            input_ids, attention_mask = tokenizer.encode(metadata["prompt"])
            encoded[index] = encoder(
                input_ids.to(device=device), attention_mask.to(device=device)
            ).cpu()

    return encoded, empty_encoding


def load_denoiser(model_path: Path, device: torch.device, dtype: torch.dtype) -> OvisImageModel:
    config = ovis_image_configs["ovis-image-7b"]
    model = OvisImageModel(config)
    state_dict = load_file(str(model_path))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing_keys}, unexpected={unexpected_keys}"
        )
    del state_dict
    gc.collect()
    return model.to(device=device, dtype=dtype).eval()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    metadatas = read_metadata(args.metadata_file)
    pending = select_pending(args, metadatas)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"GenEval prompts in metadata: {len(metadatas)}")
    print(f"Shard: {args.shard_id + 1}/{args.num_shards}")
    print(f"Pending prompts in this shard: {len(pending)}")
    print(f"Images per prompt: {args.n_samples}")
    print(f"Pending images at most: {len(pending) * args.n_samples}")

    if not pending:
        print("Nothing to generate; every expected sample already exists.")
        return

    tokenizer = build_ovis_tokenizer(str(args.ovis_path))
    print("Loading Ovis text encoder")
    encoder = OvisEmbedder(

        model_path=str(args.ovis_path),
        random_init=False,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(device=device, dtype=dtype).eval()

    prompt_encodings, empty_encoding = encode_prompts(
        pending, tokenizer, encoder, device
    )
    encoder.to("cpu")
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    print("Prompt encoding complete; text encoder released")

    print("Loading Ovis-Image denoiser")
    model = load_denoiser(args.model_path, device, dtype)
    model_config = ovis_image_configs["ovis-image-7b"]
    print("Loading VAE")
    autoencoder = load_ae(
        str(args.vae_path),
        model_config.autoencoder_params,
        device=device,
        dtype=dtype,
        random_init=False,
    ).eval()
    print("Denoiser and VAE ready")

    empty_encoding = empty_encoding.to(device=device, dtype=dtype)
    total_generated = 0
    started = time.monotonic()

    with torch.inference_mode():
        for index, metadata in tqdm(pending, desc="GenEval prompts"):
            prompt_dir = args.outdir / f"{index:05d}"
            sample_dir = prompt_dir / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            with (prompt_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False)

            condition = prompt_encodings.pop(index).to(device=device, dtype=dtype)
            for sample_index, sample_path in enumerate(
                expected_sample_paths(args.outdir, index, args.n_samples)
            ):
                if sample_path.is_file() and not args.overwrite:
                    continue

                latents = generate_noise_latent(
                    condition.shape[0],
                    args.image_size,
                    args.image_size,
                    device,
                    dtype,
                    seed=args.seed + sample_index,
                    latent_channel=autoencoder.params.z_channels,
                )
                image_latents = denoise(
                    device=device,
                    dtype=dtype,
                    model=model,
                    latents=latents,
                    denoising_steps=args.denoising_steps,
                    ovis_encodings=condition,
                    enable_classifier_free_guidance=True,
                    empty_ovis_encodings=empty_encoding,
                    classifier_free_guidance_scale=args.cfg_scale,
                )
                image = autoencoder.decode(image_latents)
                save_image(
                    name=sample_path.name,
                    output_dir=str(sample_dir),
                    x=image,
                    add_sampling_metadata=True,
                    prompt=metadata["prompt"],
                    verbose=False,
                )
                total_generated += 1

                del latents, image_latents, image

            del condition

    elapsed = time.monotonic() - started
    print(f"Generated images this run: {total_generated}")
    print(f"Generation time: {elapsed / 3600:.2f} hours")
    if total_generated:
        print(f"Average seconds per image: {elapsed / total_generated:.2f}")
    print(f"GenEval images saved under: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
