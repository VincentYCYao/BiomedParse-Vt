import argparse
import os
import json
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download

from inference_utils.inference import interactive_infer_image
from modeling import build_model
from modeling.BaseModel import BaseModel
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from utilities.distributed import init_distributed  # changed from utils

# ======
# Datasets
datasets = [
    "ACDC",
    "amos22/CT",
    "amos22/MRI",
    "BreastUS",
    "CAMUS",
    "COVID-19_CT",
    "COVID-QU-Ex",
    "CXR_Masks_and_Labels",
    "FH-PS-AOP",
    "GlaS",
    "LGG",
    "LIDC-IDRI",
    "MSD/Task01_BrainTumour",
    "MSD/Task02_Heart",
    "MSD/Task03_Liver",
    "MSD/Task04_Hippocampus",
    "MSD/Task05_Prostate",
    "MSD/Task06_Lung",
    "MSD/Task07_Pancreas",
    "MSD/Task08_HepaticVessel",
    "MSD/Task09_Spleen",
    "MSD/Task10_Colon",
    "NeoPolyp",
    "OCT-CME",
    "PanNuke",
    "Radiography/COVID",
    "Radiography/Lung_Opacity",
    "Radiography/Normal",
    "Radiography/Viral_Pneumonia",
    "REFUGE",
    "UWaterlooSkinCancer",
]
datafolder = "/mnt/vincent-pvc/Datasets/BiomedParseData"
# ======

split = "test"


def overlay_masks(image, masks, colors):
    overlay = image.copy()
    overlay = np.array(overlay, dtype=np.uint8)
    for mask, color in zip(masks, colors):
        overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array(color) * 0.6).astype(
            np.uint8
        )
    return Image.fromarray(overlay)


def overlay_contours(image, masks, colors, thickness=2):
    # Make a copy of the input image
    overlay = image.copy()

    # Ensure the image is in uint8 format
    overlay = np.array(overlay, dtype=np.uint8)

    # Iterate over each mask and corresponding color
    for index, (mask, color) in enumerate(zip(masks, colors)):
        # Find contours of the mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw all contours for this mask on the overlay
        for contour in contours:
            cv2.drawContours(overlay, [contour], -1, color, thickness)

    # Convert to PIL Image for consistency with your original function
    return Image.fromarray(overlay)


def generate_colors(n):
    cmap = plt.get_cmap("tab10")
    colors = [tuple(int(255 * val) for val in cmap(i)[:3]) for i in range(n)]
    return colors


for dataset in datasets:
    metric_dice = {}
    metric_dice["DSC"] = list()
    with open(os.path.join(datafolder, dataset, split + ".json"), "r") as f:
        datajson = json.load(f)

    # output folder
    outfolder_root = "/mnt/vincent-pvc/BiomedParse-Vt/Results_inference"
    out_fig_folder = os.path.join(outfolder_root, "figure", dataset)
    out_mask_folder = os.path.join(outfolder_root, "mask", dataset)
    os.makedirs(out_fig_folder, exist_ok=True)
    os.makedirs(out_mask_folder, exist_ok=True)

    # List of images
    datalist = datajson["annotations"]

    model_file = hf_hub_download(
        repo_id="microsoft/BiomedParse",
        filename="biomedparse_v1.pt",
        local_dir="pretrained",
    )

    print(f"Downloaded model file to: {model_file}")

    conf_files = "configs/biomedparse_inference.yaml"
    opt = load_opt_from_config_files([conf_files])
    opt = init_distributed(opt)

    model_file = "./pretrained/biomedparse_v1.pt"

    model = BaseModel(opt, build_model(opt)).from_pretrained(model_file).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )

    for i_data, data in enumerate(datalist):
        metric_dice["DSC"].append({})

        # Annotations:
        num_pixels = data["area"]
        mask_file = data["mask_file"]
        img_file = data["file_name"]

        prompts = [data["sentences"][0]["sent"]]

        # RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.
        image = Image.open(os.path.join(datafolder, dataset, split, img_file))
        image = image.convert("RGB")

        pred_mask = interactive_infer_image(model, image, prompts)
        pred_mask.shape

        # load ground truth mask
        gt_masks = []
        for prompt in prompts:
            gt_mask = Image.open(
                os.path.join(datafolder, dataset, split + "_mask", mask_file)
            )
            gt_mask = 1 * (np.array(gt_mask.convert("RGB"))[:, :, 0] > 0)
            gt_masks.append(gt_mask)

        # prediction with ground truth mask
        for i_mask, pred in enumerate(pred_mask):
            gt = gt_masks[i_mask]
            dice = (
                (1 * (pred > 0.5) & gt).sum()
                * 2.0
                / (1 * (pred > 0.5).sum() + gt.sum())
            )
            print(f"Dice score for {prompts[i_mask]}: {dice:.4f}")

        colors = generate_colors(len(prompts))

        pred_overlay = overlay_contours(
            image, [1 * (pred_mask[i] > 0.5) for i in range(len(prompts))], colors
        )

        gt_overlay = overlay_contours(image, gt_masks, colors)

        legend_patches = [
            mpatches.Patch(color=np.array(color) / 255, label=prompt)
            for color, prompt in zip(colors, prompts)
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(pred_overlay)
        axes[1].set_title("Predictions")
        axes[1].axis("off")
        axes[1].legend(handles=legend_patches, loc="upper right", fontsize="small")

        axes[2].imshow(gt_overlay)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")
        axes[2].legend(handles=legend_patches, loc="upper right", fontsize="small")

        plt.tight_layout()
        plt.savefig(os.path.join(out_fig_folder, img_file))
        plt.close(fig)

        # save the mask
        pred_mask_file = os.path.join(out_mask_folder, img_file)
        # Image.fromarray((np.squeeze(pred_mask) > 0.5)).astype(np.uint8).save(pred_mask_file)
        # Ensure the array is converted to uint8 before creating the image
        Image.fromarray(((np.squeeze(pred_mask) > 0.5) * 255).astype(np.uint8)).save(pred_mask_file)

        metric_dice["DSC"][i_data]["mask_file"] = mask_file
        metric_dice["DSC"][i_data]["img_file"] = img_file
        metric_dice["DSC"][i_data]["prompt"] = prompt
        metric_dice["DSC"][i_data]["dice"] = dice

    # Write dictionary to JSON file
    json_folder = os.path.join(outfolder_root, "metric", dataset)
    os.makedirs(json_folder, exist_ok=True)
    with open(os.path.join(json_folder, "dice.json"), "w") as file:
        json.dump(metric_dice, file, indent=4)
