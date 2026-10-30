"""
Master Pipeline Orchestrator.
Coordinates the batch-first execution lifecycle across all stages with explicit VRAM management.
"""

from pathlib import Path
from typing import List
import cv2
import numpy as np
from PIL import Image

from .blending import composite_panorama, save_final_panorama
from .config.schema import PipelineConfig
from .core.env_setup import setup_dependencies
from .core.logger import get_logger, setup_logger
from .core.path_manager import JobPaths, discover_jobs
from .geometry import (
    CUBE_FACE_KEYS,
    CubemapTransformResult,
    cubemap_to_equirectangular,
    equirectangular_to_cubemap,
    forward_geometry,
    get_active_faces,
    inverse_geometry,
)
from .inpainting import load_sd_pipeline, run_inpaint_inference, unload_sd_pipeline
from .segmentation import (
    dilate_mask,
    load_sam_model,
    run_multi_pass_segmentation,
    unload_sam_model,
)


def run_pipeline(config: PipelineConfig) -> None:
    """
    Executes the Matterport Defurnishing Pipeline according to the configured stage and parameters.
    """
    logger = setup_logger(config.log_level)
    logger.info("=" * 70)
    logger.info("Starting Matterport Defurnishing Pipeline (CVPRW 2024 Replication)")
    logger.info(f"Stage: {config.stage} | Log Level: {config.log_level}")
    logger.info("=" * 70)

    # 1. Environment & Dependency Discovery
    setup_dependencies()

    if config.hf_key:
        import os
        os.environ["HF_TOKEN"] = config.hf_key
        os.environ["HUGGING_FACE_HUB_TOKEN"] = config.hf_key

    if not config.input:
        raise ValueError("No input image or directory specified. Use --input <path>.")

    # 2. Discover Jobs
    jobs: List[JobPaths] = discover_jobs(
        input_path_str=config.input,
        output_root_str=config.output,
        recursive=config.recursive,
        output_format=config.geometry.output_format,
    )

    if not jobs:
        logger.warning("No jobs found to process. Exiting.")
        return

    # Filter out completed jobs if overwrite is disabled
    pending_jobs = jobs
    if not config.overwrite and config.stage == "all":
        pending_jobs = [j for j in jobs if not j.final_output_path.exists()]
        skipped = len(jobs) - len(pending_jobs)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already completed jobs (use --overwrite to force re-run).")

    if not pending_jobs:
        logger.info("All discovered jobs are already complete.")
        return

    # =========================================================================
    # STAGE 1: SAM 3.1 MASK GENERATION (Batch-First GPU Lifecycle)
    # =========================================================================
    if config.stage in ("all", "masks-only"):
        logger.info("--- STAGE 1: Multi-Pass SAM 3.1 Mask Generation ---")
        sam_model, sam_processor = load_sam_model(config.sam)

        try:
            for idx, job in enumerate(pending_jobs, start=1):
                final_mask_path = job.masks_dir / "final_mask.png"
                if final_mask_path.exists() and not config.overwrite:
                    logger.info(f"[{idx}/{len(pending_jobs)}] Mask already exists for {job.image_path.name}. Skipping.")
                    continue

                logger.info(f"[{idx}/{len(pending_jobs)}] Generating masks for: {job.image_path.name}")
                union_mask = run_multi_pass_segmentation(
                    model=sam_model,
                    processor=sam_processor,
                    image_path=job.image_path,
                    masks_output_dir=job.masks_dir,
                    config=config.sam,
                )

                # Dilation to cover contact shadows
                dilated_mask = dilate_mask(union_mask, dilate_pixels=config.sam.mask_dilate)
                cv2.imwrite(str(final_mask_path), dilated_mask)
                logger.debug(f"Saved final dilated mask to: {final_mask_path}")

        finally:
            # Always ensure SAM is freed from GPU memory before proceeding
            unload_sam_model(sam_model, sam_processor)

    # =========================================================================
    # STAGES 2, 3 & 4: GEOMETRY, DIFFUSION INPAINTING & BLENDING
    # =========================================================================
    if config.stage in ("all", "inpaint-only"):
        # STAGE 2: CPU Geometry Pre-processing
        if config.projection == "cubemap":
            logger.info(f"--- STAGE 2: Panorama Geometry Pre-processing (Cubemap Perspective, face_size={config.cubemap.face_size}) ---")
        else:
            logger.info("--- STAGE 2: Panorama Geometry Pre-processing (Roll & Wrap-Pad) ---")

        geom_results = {}
        for idx, job in enumerate(pending_jobs, start=1):
            job.ensure_directories()
            mask_path = job.masks_dir / "final_mask.png"
            if not mask_path.exists():
                logger.error(f"Missing required mask {mask_path} for {job.image_path.name}. Skipping inpaint.")
                continue

            orig_img = np.array(Image.open(job.image_path).convert("RGB"))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = orig_img.shape[:2]

            if config.projection == "cubemap":
                face_size = config.cubemap.face_size
                img_faces = equirectangular_to_cubemap(orig_img, face_size=face_size)
                mask_faces = equirectangular_to_cubemap(mask, face_size=face_size, is_mask=True)
                active_faces = get_active_faces(mask_faces, min_mask_pixels=20)

                cube_dir = job.intermediate_dir / "cubemap"
                cube_dir.mkdir(parents=True, exist_ok=True)
                for k in CUBE_FACE_KEYS:
                    cv2.imwrite(str(cube_dir / f"face_{k}.png"), cv2.cvtColor(img_faces[k], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(cube_dir / f"mask_{k}.png"), mask_faces[k])

                logger.info(f"[{idx}/{len(pending_jobs)}] {job.image_path.name} active faces needing inpainting: {active_faces or 'None'}")
                geom_results[job.image_path] = CubemapTransformResult(
                    img_faces=img_faces,
                    mask_faces=mask_faces,
                    active_faces=active_faces,
                    orig_width=orig_w,
                    orig_height=orig_h,
                )
            else:
                result = forward_geometry(orig_img, mask, config.geometry)
                geom_results[job.image_path] = result

                # Save intermediate outputs
                cv2.imwrite(
                    str(job.intermediate_dir / "rolled_padded_image.png"),
                    cv2.cvtColor(result.transformed_image, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(job.intermediate_dir / "rolled_padded_mask.png"),
                    result.transformed_mask,
                )
                result.save_meta(job.intermediate_dir / "transform_meta.json")

        # STAGE 3: Stable Diffusion Inpainting (Loaded ONCE in VRAM)
        logger.info("--- STAGE 3: Stable Diffusion Inpainting ---")
        sd_pipe = load_sd_pipeline(config.sd)

        try:
            for idx, job in enumerate(pending_jobs, start=1):
                if job.image_path not in geom_results:
                    continue

                geom_res = geom_results[job.image_path]

                if config.projection == "cubemap":
                    logger.info(f"[{idx}/{len(pending_jobs)}] Inpainting Cubemap Faces: {job.image_path.name}")
                    cube_dir = job.intermediate_dir / "cubemap"
                    for face_key in CUBE_FACE_KEYS:
                        if face_key in geom_res.active_faces:
                            logger.info(f"  Inpainting Face {face_key}...")
                            inpaint_face = run_inpaint_inference(
                                pipe=sd_pipe,
                                image_prepared=geom_res.img_faces[face_key],
                                mask_prepared=geom_res.mask_faces[face_key],
                                config=config.sd,
                            )
                            cv2.imwrite(
                                str(cube_dir / f"inpaint_{face_key}.png"),
                                cv2.cvtColor(inpaint_face, cv2.COLOR_RGB2BGR),
                            )
                            geom_res.inpainted_faces[face_key] = inpaint_face
                        else:
                            geom_res.inpainted_faces[face_key] = geom_res.img_faces[face_key]
                else:
                    logger.info(f"[{idx}/{len(pending_jobs)}] Inpainting Equirectangular: {job.image_path.name}")
                    inpaint_raw = run_inpaint_inference(
                        pipe=sd_pipe,
                        image_prepared=geom_res.transformed_image,
                        mask_prepared=geom_res.transformed_mask,
                        config=config.sd,
                    )

                    # Save raw inpaint
                    cv2.imwrite(
                        str(job.intermediate_dir / "inpaint_raw.png"),
                        cv2.cvtColor(inpaint_raw, cv2.COLOR_RGB2BGR),
                    )
                    # Attach raw inpaint to job for blending stage
                    geom_res.inpaint_raw = inpaint_raw

        finally:
            # Always ensure SD is freed from GPU memory
            unload_sd_pipeline(sd_pipe)

        # STAGE 4: CPU Blending & High-Frequency Compositing
        logger.info("--- STAGE 4: Post-Processing & Compositing ---")
        for idx, job in enumerate(pending_jobs, start=1):
            if job.image_path not in geom_results:
                continue

            geom_res = geom_results[job.image_path]
            orig_img = np.array(Image.open(job.image_path).convert("RGB"))
            final_mask = cv2.imread(str(job.masks_dir / "final_mask.png"), cv2.IMREAD_GRAYSCALE)

            if config.projection == "cubemap":
                if not geom_res.inpainted_faces:
                    continue
                restored_inpaint = cubemap_to_equirectangular(
                    faces=geom_res.inpainted_faces,
                    out_width=geom_res.orig_width,
                    out_height=geom_res.orig_height,
                )
                cv2.imwrite(
                    str(job.intermediate_dir / "restored_inpaint_pano.png"),
                    cv2.cvtColor(restored_inpaint, cv2.COLOR_RGB2BGR),
                )
            else:
                if not hasattr(geom_res, "inpaint_raw"):
                    continue
                # Invert geometry (Upsample -> Unpad -> Unroll)
                restored_inpaint = inverse_geometry(geom_res.inpaint_raw, geom_res.to_dict())

            # Alpha composite with original image
            final_composite = composite_panorama(
                original_image=orig_img,
                restored_inpaint=restored_inpaint,
                binary_mask=final_mask,
                feather_radius=config.geometry.blend_feather,
            )

            # Save final output
            save_final_panorama(
                image=final_composite,
                output_path=job.final_output_path,
                output_format=config.geometry.output_format,
                jpeg_quality=config.geometry.jpeg_quality,
            )
            logger.info(f"[{idx}/{len(pending_jobs)}] Saved final defurnished panorama: {job.final_output_path}")

    logger.info("=" * 70)
    logger.info("Defurnishing Pipeline execution successfully finished.")
    logger.info("=" * 70)


def cli_main() -> None:
    """CLI entry point for installed console script."""
    from .config.loader import load_config_and_args
    config = load_config_and_args()
    run_pipeline(config)


if __name__ == "__main__":
    cli_main()
