import json
import os

import torch
import torchvision.models as models
from torchvision.transforms import v2 as transforms

from utils.CustomTransforms import SquarePad

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

NUM_TOP_CLASSES = 5

MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')


def discover_models():
    models_found = {}
    for name in sorted(os.listdir(MODELS_DIR), reverse=True):
        model_dir = os.path.join(MODELS_DIR, name)
        if (os.path.isdir(model_dir)
                and os.path.isfile(os.path.join(model_dir, 'classes.txt'))
                and os.path.isfile(os.path.join(model_dir, 'weights.pth'))):
            display_name = name.replace('-', ' ')
            models_found[display_name] = name
    return models_found


AVAILABLE_MODELS = discover_models()
DEFAULT_MODEL = next(iter(AVAILABLE_MODELS)) if AVAILABLE_MODELS else None

_loaded_models = {}


def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip().strip("'").strip() for line in f if line.strip()]


def build_resnet50(num_classes, weights_path):
    net = models.resnet50()
    net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()
    return net.to(DEVICE)


image_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    SquarePad(),
    transforms.Resize((224, 224), antialias=True),
])


def format_label(name):
    return name.replace('_', ' ')


def load_thresholds(path, labels):
    if not os.path.exists(path):
        return {}, {}
    with open(path, 'r') as f:
        data = json.load(f)
    thresholds = {}
    for idx_str, metrics in data.get("class_metrics", {}).items():
        idx = int(idx_str)
        if idx < len(labels):
            thresholds[labels[idx]] = metrics["threshold"]
    meta = {
        k: v for k, v in data.items()
        if k != "class_metrics"
    }
    return thresholds, meta


def get_model(name=None):
    if name is None:
        name = DEFAULT_MODEL
    if name in _loaded_models:
        return _loaded_models[name]

    dir_name = AVAILABLE_MODELS[name]
    model_dir = os.path.join(BASE_DIR, 'data', 'models', dir_name)

    labels = load_labels(os.path.join(model_dir, 'classes.txt'))
    net = build_resnet50(len(labels), os.path.join(model_dir, 'weights.pth'))
    thresholds, threshold_meta = load_thresholds(
        os.path.join(model_dir, 'thresholds.json'), labels
    )
    num_params = sum(p.numel() for p in net.parameters())

    _loaded_models[name] = (labels, net, thresholds, threshold_meta, num_params)
    return _loaded_models[name]


def render_predictions(predictions, model_name=None):
    if not predictions:
        return '<div class="pred-panel"><p class="pred-empty">No predictions</p></div>'

    if model_name:
        _, _, class_thresholds, _, _ = get_model(model_name)
    else:
        class_thresholds = {}

    sorted_preds = sorted(
        predictions.items(), key=lambda x: x[1], reverse=True
    )[:NUM_TOP_CLASSES]

    model_subtitle = ""
    if model_name:
        model_subtitle = (
            f'<div class="pred-model-name"'
            f' style="font-size:0.85em;opacity:0.6;margin-bottom:8px">'
            f'{model_name}</div>'
        )

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

    return '<div class="pred-panel">' + model_subtitle + ''.join(rows) + legend + '</div>'


async def predict(image, model_name=None):
    if image is None:
        return {}

    labels, net, _, _, _ = get_model(model_name)

    image_rgb = image.convert('RGB')
    tensor = image_transform(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = net(tensor)[0]
        probs = torch.nn.functional.softmax(logits, dim=0)

    return {
        labels[i]: float(probs[i])
        for i in range(len(labels))
    }


async def predict_scores(image, model_name=None):
    """Return all class scores as a JSON-serialisable dict.

    Returns:
        {"class_labels": [...], "scores": [...]}
    where scores are ordered to match class_labels.
    """
    preds = await predict(image, model_name=model_name)
    if not preds:
        return {"class_labels": [], "scores": []}
    labels, _, _, _, _ = get_model(model_name)
    ordered_labels = list(labels)
    ordered_scores = [preds.get(label, 0.0) for label in ordered_labels]
    return {"class_labels": ordered_labels, "scores": ordered_scores}


async def predict_html(image, model_name=None):
    preds = await predict(image, model_name=model_name)
    return render_predictions(preds, model_name=model_name)


def get_thresholds(model_name=None):
    """Return per-class thresholds and class labels for a model.

    Returns a JSON-serialisable dict with:
      - class_labels: list of class names (same order as score columns)
      - thresholds: dict mapping class name -> optimal F2 threshold
      - model_name: display name passed in (or the default)
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    labels, _, thresholds, _, _ = get_model(model_name)
    return {
        "class_labels": list(labels),
        "thresholds": thresholds,
        "model_name": model_name,
    }


def build_about_markdown(model_name=None):
    labels, _, _, threshold_meta, num_params = get_model(model_name)
    class_list = "\n".join(
        f"1. {format_label(name)}" for name in labels
    )
    meta_model_name = threshold_meta.get("model_name", "")
    model_name_line = f"- **Model name:** {meta_model_name}\n" if meta_model_name else ""
    return (
        "## Model\n\n"
        f"{model_name_line}"
        f"- **Architecture:** ResNet-50\n"
        f"- **Parameters:** {num_params:,}\n"
        f"- **Input size:** 224 x 224 (square-padded)\n"
        f"- **Classes:** {len(labels)}\n\n"
        "## Training data\n\n"
        "Fine-tuned using image data from the "
        "[SMHI IFCB Plankton Image Reference Library]"
        "(https://doi.org/10.17044/scilifelab.25883455) "
        "and images provided by "
        "The Norwegian Institute for Water Research (NIVA).\n\n"
        "## Class list\n\n"
        f"{class_list}\n"
    )
