# https://huggingface.co/lch01/StreamVGGT

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import cv2
import torch
import numpy as np
import sys
import shutil
import glob
import gc
import time

sys.path.append("src/")
from visual_util import predictions_to_glb
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    predictions = {}
    predictions["images"] = images  # (S, 3, H, W)
    print(f"Images shape: {images.shape}")

    frames = []
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0)
        frame = {
            "img": image
        }
        frames.append(frame)

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            output = model.inference(frames)

    all_pts3d = []
    all_conf = []
    all_depth = []
    all_depth_conf = []
    all_camera_pose = []

    for res in output.ress:
        all_pts3d.append(res['pts3d_in_other_view'].squeeze(0))
        all_conf.append(res['conf'].squeeze(0))
        all_depth.append(res['depth'].squeeze(0))
        all_depth_conf.append(res['depth_conf'].squeeze(0))
        all_camera_pose.append(res['camera_pose'].squeeze(0))

    predictions["world_points"] = torch.stack(all_pts3d, dim=0)  # (S, H, W, 3)
    predictions["world_points_conf"] = torch.stack(all_conf, dim=0)  # (S, H, W)
    predictions["depth"] = torch.stack(all_depth, dim=0)  # (S, H, W, 1)
    predictions["depth_conf"] = torch.stack(all_depth_conf, dim=0)  # (S, H, W)
    predictions["pose_enc"] = torch.stack(all_camera_pose, dim=0)  # (S, 9)

    print("World points shape:", predictions["world_points"].shape)
    print("World points confidence shape:", predictions["world_points_conf"].shape)
    print("Depth map shape:", predictions["depth"].shape)
    print("Depth confidence shape:", predictions["depth_conf"].shape)
    print("Pose encoding shape:", predictions["pose_enc"].shape)
    print(f"Images shape: {images.shape}")

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic.squeeze(0)  # (S, 3, 4)
    predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None  # (S, 3, 3) or None
    print("Extrinsic shape:", predictions["extrinsic"].shape)
    print("Intrinsic shape:", predictions["intrinsic"].shape)

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy()

    predictions["world_points_from_depth"] = predictions["world_points"]

    # Clean up
    torch.cuda.empty_cache()
    return predictions

def main():
    """
    Main function to run the simplified demo.
    """
    print("Initializing and loading StreamVGGT model...")

    # import pdb; pdb.set_trace()

    local_ckpt_path = "./ckpt/model.pt"
    if os.path.exists(local_ckpt_path):
        print(f"Loading local checkpoint from {local_ckpt_path}")
        model = StreamVGGT()
        ckpt = torch.load(local_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        del ckpt
    else:
        print("Local checkpoint not found, downloading from Hugging Face...")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="lch01/StreamVGGT",
            filename="checkpoints.pth",
            revision="main",
            force_download=True
        )
        model = StreamVGGT()
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        del ckpt

    # 1. Define input and output paths
    example_image_dir = "examples/example_building/"
    output_dir = "output/example_building_result"
    output_images_dir = os.path.join(output_dir, "images")

    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    os.makedirs(output_images_dir, exist_ok=True)

    # 2. Copy example images to the target directory structure
    image_files = glob.glob(os.path.join(example_image_dir, "*.jpg"))
    for img_path in image_files:
        shutil.copy(img_path, os.path.join(output_images_dir, os.path.basename(img_path)))

    print(f"Copied {len(image_files)} images to {output_images_dir}")

    # 3. Run the model
    start_time = time.time()
    print("Running model...")
    predictions = run_model(output_dir, model)

    # 4. Save predictions and generate .glb file
    prediction_save_path = os.path.join(output_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)
    print(f"Predictions saved to {prediction_save_path}")

    glbfile = os.path.join(output_dir, "scene.glb")

    print(f"Generating GLB file at {glbfile}...")
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=50.0,
        filter_by_frames="All",
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=True,
        mask_sky=False,
        target_dir=output_dir,
        prediction_mode="Depthmap and Camera Branch",
    )
    glbscene.export(file_obj=glbfile)
    print("GLB file generated successfully.")

    # Clean up
    del predictions
    gc.collect()
    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print("Done.")


if __name__ == "__main__":
    main()