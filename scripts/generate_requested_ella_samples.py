#!/usr/bin/env python3
"""Generate the four requested portrait samples with the trained ELLA connector."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)

from train import (
    TrainingState,
    load_clip,
    load_gemma,
    load_stylejourney,
    make_encode_clip,
    make_encode_gemma,
)
from pure_ella.config import TrainConfig, seed_everything
from pure_ella.diagnostics import (
    generate_clip_teacher,
    generate_ella,
    save_validation_grid,
)
from scripts.compare_phase0_connectors import load_connector


CONFIG = "config_phase0_ella_tsc_full77_randomt_b64_250k.json"
CHECKPOINT = (
    "output_phase0_ella_tsc_full77_randomt_b64_250k/"
    "pure_ella_connector_L77.pt"
)
STYLEJOURNEY = (
    "/mnt/e/stable-diffusion-webui/models/Stable-diffusion/"
    "stylejourney_15_nice-000004.safetensors"
)
OUTPUT_DIR = Path("output_requested_ella_250k")


@dataclass(frozen=True)
class Sample:
    name: str
    prompt: str
    negative_prompt: str
    steps: int
    guidance: float
    sampler: str
    seed: int
    width: int = 768
    height: int = 960


SAMPLES = (
    Sample(
        name="warrior_princess",
        prompt=(
            "poster of warrior princess| standing on hill | centered| key "
            "visual| intricate| highly detailed| breathtaking| precise "
            "lineart| vibrant| panoramic| cinematic| Carne Griffiths| "
            "Conrad Roset"
        ),
        negative_prompt=(
            "(bonnet), (hat), (beanie), cap, (((wide shot))), (cropped "
            "head), bad framing, out of frame, deformed, cripple, old, fat, "
            "ugly, poor, missing arm, additional arms, additional legs, "
            "additional head, additional face, multiple people, group of "
            "people, dyed hair, black and white, grayscale"
        ),
        steps=30,
        guidance=7.0,
        sampler="dpmpp_sde_karras",
        seed=4267154965,
    ),
    Sample(
        name="medieval_darth_vader_dragon",
        prompt=(
            "photo of medieval darth vader on red dragon, epic fight scene, "
            "holding glowing red flaming sword, full body, hyper realistic, "
            "intricate, apocalyptic"
        ),
        negative_prompt="dark",
        steps=20,
        guidance=7.0,
        sampler="euler_a",
        seed=744448957,
    ),
    Sample(
        name="goddess_of_death",
        prompt=(
            "Goddess, Goddess of Death, detailing, facial detailing, ultra "
            "quality, cinematic lighting, perfect and beautiful face, "
            "perfect composition, realistic, circuit board, fantasy, "
            "illustration, artstation, trial dark fantasy, photorealistic "
            "concept art, intense shadows, intense lighting :: 8k resolution, "
            "ultra-detailed quality 3D octane render, sharp focus, wallpaper, "
            "HDR, high quality, high-definition stylize 500"
        ),
        negative_prompt=(
            "(deformed mouth), (deformed lips), (deformed eyes), "
            "(cross-eyed), (deformed iris), (deformed hands), lowers, 3d "
            "render, cartoon, long body, wide hips, narrow waist, disfigured, "
            "ugly, cross eyed, squinting, grain, Deformed, blurry, bad "
            "anatomy, poorly drawn face, mutation, mutated, extra limb, ugly, "
            "(poorly drawn hands), missing limb, floating limbs, disconnected "
            "limbs, malformed hands, blur, out of focus, long neck, "
            "disgusting, poorly drawn, mutilated, mangled, old, surreal, "
            "((text)), jewelery, earrings"
        ),
        steps=30,
        guidance=7.0,
        sampler="dpmpp_sde_karras",
        seed=1826311933,
    ),
    Sample(
        name="milkyway",
        prompt="milkyway",
        negative_prompt="dark",
        steps=30,
        guidance=7.0,
        sampler="dpmpp_sde_karras",
        seed=1826311933,
    ),
)


def make_scheduler(name: str, training_scheduler):
    if name == "dpmpp_sde_karras":
        return DPMSolverMultistepScheduler.from_config(
            training_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
        )
    if name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(
            training_scheduler.config)
    raise ValueError(f"Unsupported sampler: {name}")


def main() -> None:
    cfg = TrainConfig.from_json(CONFIG)
    cfg.sd_checkpoint = STYLEJOURNEY
    cfg.enable_unet_gradient_checkpointing = False
    seed_everything(cfg.base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    state = TrainingState(
        cfg=cfg,
        device=device,
        unet_dtype=dtype,
        autocast_dtype=dtype if device.type == "cuda" else None,
    )
    load_gemma(state)
    load_clip(state)
    load_stylejourney(state)
    state.encode_gemma = make_encode_gemma(state)
    state.encode_clip = make_encode_clip(state)
    state.connector = load_connector(CONFIG, CHECKPOINT, state)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = []
    labels = []
    for sample in SAMPLES:
        state.inf_scheduler = make_scheduler(sample.sampler, state.scheduler)
        ella_image = generate_ella(
            sample.prompt,
            state,
            steps=sample.steps,
            guidance=sample.guidance,
            seed=sample.seed,
            negative_prompt=sample.negative_prompt,
            height=sample.height,
            width=sample.width,
        )
        ella_path = OUTPUT_DIR / f"{sample.name}_ella.png"
        ella_image.save(ella_path)
        images.append(ella_image)
        labels.append(
            f"ELLA 250k | {sample.name} | {sample.sampler} | "
            f"{sample.steps} steps | "
            f"CFG {sample.guidance:g} | seed {sample.seed}"
        )
        print(f"Saved {ella_path.resolve()}")

        state.inf_scheduler = make_scheduler(sample.sampler, state.scheduler)
        clip_image = generate_clip_teacher(
            sample.prompt,
            state,
            steps=sample.steps,
            guidance=sample.guidance,
            seed=sample.seed,
            negative_prompt=sample.negative_prompt,
            height=sample.height,
            width=sample.width,
        )
        clip_path = OUTPUT_DIR / f"{sample.name}_clip.png"
        clip_image.save(clip_path)
        images.append(clip_image)
        labels.append(
            f"CLIP | {sample.name} | {sample.sampler} | "
            f"{sample.steps} steps | CFG {sample.guidance:g} | "
            f"seed {sample.seed}"
        )
        print(f"Saved {clip_path.resolve()}")

    grid_path = OUTPUT_DIR / "requested_samples_ella_clip_grid.png"
    save_validation_grid(
        images, labels, str(grid_path),
        "ELLA full77 250k vs CLIP + StyleJourney",
        cols=2,
    )
    print(f"Saved {grid_path.resolve()}")


if __name__ == "__main__":
    main()
