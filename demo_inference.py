import argparse
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from inference_utils.inference import interactive_infer_image
from modeling import build_model
from modeling.BaseModel import BaseModel
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from utilities.distributed import init_distributed  # changed from utils

model_file = hf_hub_download(repo_id="microsoft/BiomedParse", filename="biomedparse_v1.pt", local_dir="pretrained", )

print(f"Downloaded model file to: {model_file}")

conf_files = "configs/biomedparse_inference.yaml"
opt = load_opt_from_config_files([conf_files])
opt = init_distributed(opt)

model_file = "./pretrained/biomedparse_v1.pt"

model = BaseModel(opt, build_model(opt)).from_pretrained(model_file).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

# RGB image input of shape (H, W, 3). Currently only batch size 1 is supported.
image = Image.open("examples/Part_1_516_pathology_breast.png", formats=["png"])
image = image.convert("RGB")

# text prompts querying objects in the image. Multiple ones can be provided.
prompts = ["neoplastic cells", "inflammatory cells"]

pred_mask = interactive_infer_image(model, image, prompts)
pred_mask.shape

# load ground truth mask
gt_masks = []
for prompt in prompts:
    gt_mask = Image.open(f"examples/Part_1_516_pathology_breast_{prompt.replace(' ', '+')}.png", formats=["png"], )
    gt_mask = 1 * (np.array(gt_mask.convert("RGB"))[:, :, 0] > 0)
    gt_masks.append(gt_mask)

# prediction with ground truth mask
for i, pred in enumerate(pred_mask):
    gt = gt_masks[i]
    dice = (1 * (pred > 0.5) & gt).sum() * 2.0 / (1 * (pred > 0.5).sum() + gt.sum())
    print(f"Dice score for {prompts[i]}: {dice:.4f}")


def overlay_masks(image, masks, colors):
    overlay = image.copy()
    overlay = np.array(overlay, dtype=np.uint8)
    for mask, color in zip(masks, colors):
        overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
    return Image.fromarray(overlay)


def generate_colors(n):
    cmap = plt.get_cmap("tab10")
    colors = [tuple(int(255 * val) for val in cmap(i)[:3]) for i in range(n)]
    return colors


original_image = Image.open("examples/Part_1_516_pathology_breast.png").convert("RGB")

colors = generate_colors(len(prompts))

pred_overlay = overlay_masks(original_image, [1 * (pred_mask[i] > 0.5) for i in range(len(prompts))], colors)

gt_overlay = overlay_masks(original_image, gt_masks, colors)

legend_patches = [mpatches.Patch(color=np.array(color) / 255, label=prompt) for color, prompt in zip(colors, prompts)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image)
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

# Save the plot to a PNG file
plt.savefig("output.png")
