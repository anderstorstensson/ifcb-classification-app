import json
import os

import numpy as np
import torch
import torchvision.models as models
from torchvision.transforms import v2 as transforms
from PIL import Image as PILImage

from utils.CustomTransforms import SquarePad

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'models', 'ifcb-plankton-resnet50')

NUM_TOP_CLASSES = 5


def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip().strip("'").strip() for line in f if line.strip()]


def build_resnet50(num_classes, weights_path):
    net = models.resnet50()
    net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net


labels = load_labels(os.path.join(MODEL_DIR, 'classes.txt'))
model = build_resnet50(len(labels), os.path.join(MODEL_DIR, 'weights.pth'))

num_params = sum(p.numel() for p in model.parameters())

image_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    SquarePad(),
    transforms.Resize((224, 224), antialias=True),
])


def format_label(name):
    return name.replace('_', ' ')


def load_thresholds(path):
    if not os.path.exists(path):
        return {}, {}
    with open(path, 'r') as f:
        data = json.load(f)
    thresholds = {}
    for idx_str, metrics in data.get("class_metrics", {}).items():
        idx = int(idx_str)
        if idx < len(labels):
            thresholds[format_label(labels[idx])] = metrics["threshold"]
    meta = {
        k: v for k, v in data.items()
        if k != "class_metrics"
    }
    return thresholds, meta


class_thresholds, threshold_meta = load_thresholds(
    os.path.join(MODEL_DIR, 'thresholds.json')
)


def render_predictions(predictions):
    if not predictions:
        return '<div class="pred-panel"><p class="pred-empty">No predictions</p></div>'

    sorted_preds = sorted(
        predictions.items(), key=lambda x: x[1], reverse=True
    )[:NUM_TOP_CLASSES]

    rows = []
    for name, prob in sorted_preds:
        threshold = class_thresholds.get(name)
        pct = prob * 100

        threshold_html = ""
        if threshold is not None:
            t_pct = threshold * 100
            threshold_html = (
                f'<div class="pred-threshold" '
                f'style="left:{t_pct:.1f}%" '
                f'title="Threshold: {threshold:.2f}"></div>'
            )

        rows.append(
            f'<div class="pred-row">'
            f'<div class="pred-header">'
            f'<span class="pred-name">{name}</span>'
            f'<span class="pred-pct">{pct:.1f}%</span>'
            f'</div>'
            f'<div class="pred-track">'
            f'<div class="pred-fill" style="width:{pct:.1f}%"></div>'
            f'{threshold_html}'
            f'</div>'
            f'</div>'
        )

    legend = (
        '<div class="pred-legend">'
        '<span class="pred-legend-marker"></span>'
        '<span>F2 threshold</span>'
        '</div>'
    )

    return '<div class="pred-panel">' + ''.join(rows) + legend + '</div>'


def stretch_contrast(image):
    arr = np.array(image, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if lo == hi:
        return image
    stretched = ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    return PILImage.fromarray(stretched, mode=image.mode)


async def predict(image, stretch=False):
    if image is None:
        return {}

    if stretch:
        image = stretch_contrast(image)

    image_rgb = image.convert('RGB')
    tensor = image_transform(image_rgb).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)[0]
        probs = torch.nn.functional.softmax(logits, dim=0)

    return {
        format_label(labels[i]): float(probs[i])
        for i in range(len(labels))
    }


async def predict_html(image, stretch=False):
    preds = await predict(image, stretch=stretch)
    return render_predictions(preds)


def build_about_markdown():
    class_list = "\n".join(
        f"1. {format_label(name)}" for name in labels
    )
    model_name = threshold_meta.get("model_name", "")
    model_name_line = f"- **Model name:** {model_name}\n" if model_name else ""
    return (
        "## Model\n\n"
        f"{model_name_line}"
        f"- **Architecture:** ResNet-50\n"
        f"- **Parameters:** {num_params:,}\n"
        f"- **Input size:** 224 x 224 (square-padded)\n"
        f"- **Classes:** {len(labels)}\n\n"
        "## Image extraction\n\n"
        "This model was trained on images extracted with "
        "[`iRfcb::ifcb_extract_pngs()`]"
        "(https://europeanifcbgroup.github.io/iRfcb/), "
        "which produces full-range [0, 255] PNGs, and expects the same format as input. "
        "Images exported by the IFCB Dashboard use a compressed pixel range "
        "and will classify poorly unless **Contrast stretch** is enabled.\n\n"
        "## Training data\n\n"
        "Fine-tuned using image data from the "
        "[SMHI IFCB Plankton Image Reference Library]"
        "(https://doi.org/10.17044/scilifelab.25883455) "
        "and images provided by "
        "The Norwegian Institute for Water Research (NIVA).\n\n"
        "## Class list\n\n"
        f"{class_list}\n"
    )
