import atexit
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import zipfile

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms import v2 as transforms
import torchvision.models as models

from utils.CustomTransforms import SquarePad

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'models', 'ifcb-plankton-resnet50')

IMAGE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp',
}

IMAGES_PER_PAGE = 500
MAX_ZIP_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_ZIP_FILES = 10_000
MAX_SINGLE_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


# --- Model loading ---


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

NUM_TOP_CLASSES = 5


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


# --- Predict ---


async def predict(image):
    if image is None:
        return {}

    image_rgb = image.convert('RGB')
    tensor = image_transform(image_rgb).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)[0]
        probs = torch.nn.functional.softmax(logits, dim=0)

    return {
        format_label(labels[i]): float(probs[i])
        for i in range(len(labels))
    }


async def predict_html(image):
    preds = await predict(image)
    return render_predictions(preds)


# --- Session helpers (disk-backed, not in-memory) ---

log = logging.getLogger(__name__)

_active_session_dirs = set()
_session_lock = threading.Lock()


@atexit.register
def _cleanup_all_sessions():
    with _session_lock:
        for d in _active_session_dirs:
            shutil.rmtree(d, ignore_errors=True)
        _active_session_dirs.clear()


def init_session(session):
    if session is None:
        session_dir = tempfile.mkdtemp(prefix="ifcb-session-")
        with _session_lock:
            _active_session_dirs.add(session_dir)
        return {"dir": session_dir, "paths": []}
    return session


def save_image(pil_image, session, original_name=""):
    session = init_session(session)
    idx = len(session["paths"])
    path = os.path.join(session["dir"], f"{idx:06d}.png")
    pil_image.save(path)
    w, h = pil_image.size
    return {
        **session,
        "paths": [*session["paths"], path],
        "sizes": [*session.get("sizes", []), w * h],
        "names": [*session.get("names", []), original_name],
    }


def extract_roi(name):
    stem = os.path.splitext(name)[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return str(int(parts[1]))
    return ""


def sorted_indices(session, sort_by_dim):
    if not session:
        return []
    n = len(session["paths"])
    if sort_by_dim:
        sizes = session.get("sizes", [])
        return sorted(range(n), key=lambda i: sizes[i] if i < len(sizes) else 0, reverse=True)
    names = session.get("names", [])
    return sorted(range(n), key=lambda i: names[i] if i < len(names) else "")


def gallery_page(session, page, sort_by_dim=True):
    if not session:
        return []
    indices = sorted_indices(session, sort_by_dim)
    names = session.get("names", [])
    paths = session["paths"]
    start = page * IMAGES_PER_PAGE
    page_indices = indices[start:start + IMAGES_PER_PAGE]
    result = []
    for i in page_indices:
        name = names[i] if i < len(names) else ""
        roi = extract_roi(name)
        caption = f"ROI {roi}" if roi else None
        result.append((paths[i], caption))
    return result


def page_count(session):
    n = len(session["paths"]) if session else 0
    return max(1, -(-n // IMAGES_PER_PAGE))


def page_info_text(session, page):
    n = len(session["paths"]) if session else 0
    if n == 0:
        return "No images uploaded"
    pages = page_count(session)
    return f"Page {page + 1} of {pages} ({n} images)"


# --- Event handlers ---


def nav_buttons(session, page):
    pages = page_count(session)
    return (
        gr.Button(interactive=(page > 0)),
        gr.Button(interactive=(page < pages - 1)),
    )


async def handle_image_upload(image, session, page, sort_by_dim):
    if image is None:
        prev, nxt = nav_buttons(session, page)
        return session, gr.Image(label="Upload image"), gallery_page(session, page, sort_by_dim), page, page_info_text(session, page), prev, nxt, render_predictions({})
    original_name = ""
    if hasattr(image, "filename") and image.filename:
        original_name = os.path.basename(image.filename)
    session = save_image(image, session, original_name)
    page = 0
    preds = await predict(image)
    prev, nxt = nav_buttons(session, page)
    label = os.path.splitext(original_name)[0] if original_name else "Upload image"
    return session, gr.Image(value=image, label=label), gallery_page(session, page, sort_by_dim), page, page_info_text(session, page), prev, nxt, render_predictions(preds)


async def handle_zip_upload(zip_path, session, page, sort_by_dim):
    if zip_path is None:
        prev, nxt = nav_buttons(session, page)
        return session, gallery_page(session, page, sort_by_dim), page, page_info_text(session, page), prev, nxt
    session = init_session(session)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            entries = [
                info for info in zf.infolist()
                if not info.is_dir()
                and os.path.splitext(os.path.basename(info.filename))[1].lower() in IMAGE_EXTENSIONS
                and not os.path.basename(info.filename).startswith('.')
            ]
            if len(entries) > MAX_ZIP_FILES:
                gr.Warning(f"ZIP contains too many files (max {MAX_ZIP_FILES}).")
                prev, nxt = nav_buttons(session, page)
                return session, gallery_page(session, page, sort_by_dim), page, page_info_text(session, page), prev, nxt
            total_read = 0
            for info in entries:
                try:
                    with zf.open(info) as f:
                        data = f.read(MAX_SINGLE_FILE_BYTES + 1)
                    if len(data) > MAX_SINGLE_FILE_BYTES:
                        continue
                    total_read += len(data)
                    if total_read > MAX_ZIP_BYTES:
                        gr.Warning("ZIP exceeds maximum total size. Some images were skipped.")
                        break
                    img = Image.open(io.BytesIO(data)).copy()
                    original_name = os.path.basename(info.filename)
                    session = save_image(img, session, original_name)
                except Exception:
                    continue
    except zipfile.BadZipFile:
        gr.Warning("The uploaded file is not a valid ZIP archive.")
    except Exception:
        log.exception("ZIP processing error")
        gr.Warning("Error processing ZIP file.")
    page = 0
    prev, nxt = nav_buttons(session, page)
    return session, gallery_page(session, page, sort_by_dim), page, page_info_text(session, page), prev, nxt


async def on_gallery_select(session, page, sort_by_dim, evt: gr.SelectData):
    indices = sorted_indices(session, sort_by_dim)
    pos = page * IMAGES_PER_PAGE + evt.index
    if pos < 0 or pos >= len(indices):
        return gr.Image(value=None, label="Upload image"), render_predictions({})
    idx = indices[pos]
    paths = session["paths"]
    names = session.get("names", [])
    img = Image.open(paths[idx])
    preds = await predict(img)
    name = names[idx] if idx < len(names) else ""
    label = os.path.splitext(name)[0] if name else "Upload image"
    return gr.Image(value=img, label=label), render_predictions(preds)


def go_prev(session, page, sort_by_dim):
    page = max(0, page - 1)
    prev, nxt = nav_buttons(session, page)
    return gallery_page(session, page, sort_by_dim), page, page_info_text(session, page), prev, nxt


def go_next(session, page, sort_by_dim):
    page = min(page_count(session) - 1, page + 1)
    prev, nxt = nav_buttons(session, page)
    return gallery_page(session, page, sort_by_dim), page, page_info_text(session, page), prev, nxt


def on_sort_change(session, sort_by_dim):
    page = 0
    prev, nxt = nav_buttons(session, page)
    return gallery_page(session, page, sort_by_dim), page, page_info_text(session, page), prev, nxt


# --- About section content ---


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
        "## Training data\n\n"
        "Fine-tuned using image data from the "
        "[SMHI IFCB Plankton Image Reference Library]"
        "(https://doi.org/10.17044/scilifelab.25883455) "
        "and images provided by "
        "The Norwegian Institute for Water Research (NIVA).\n\n"
        "## Class list\n\n"
        f"{class_list}\n"
    )


# --- UI ---

css = """
.main-title {
    text-align: center;
    margin-bottom: 0.25em;
}
.subtitle {
    text-align: center;
    opacity: 0.7;
    margin-top: 0;
    font-size: 1.05em;
}
.image-lightbox {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.85);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    cursor: pointer;
}
.image-lightbox img {
    max-width: 90vw;
    max-height: 90vh;
    object-fit: contain;
    image-rendering: pixelated;
}
.page-info {
    text-align: center;
    opacity: 0.7;
    font-size: 0.95em;
}
#sort-row {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 0 !important;
    max-width: 300px !important;
    margin: 0 auto !important;
}
#sort-row > div {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    white-space: nowrap !important;
}
#sort-toggle {
    padding: 0 !important;
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    min-width: 0 !important;
}
#sort-toggle .label-wrap { display: none !important; }
#sort-toggle input[type="checkbox"] {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 44px !important;
    height: 24px !important;
    background: #ccc !important;
    border-radius: 12px !important;
    position: relative !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
    margin: 0 !important;
}
#sort-toggle input[type="checkbox"]::after {
    content: "" !important;
    position: absolute !important;
    top: 2px !important;
    left: 2px !important;
    width: 20px !important;
    height: 20px !important;
    background: white !important;
    border-radius: 50% !important;
    transition: transform 0.2s !important;
}
#sort-toggle input[type="checkbox"]:checked {
    background: var(--color-accent) !important;
}
#sort-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px) !important;
}
.pred-panel {
    padding: 12px 0;
}
.pred-empty {
    opacity: 0.5;
    text-align: center;
    padding: 2em 0;
}
.pred-row {
    margin-bottom: 10px;
}
.pred-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
    font-size: 0.9em;
}
.pred-name {
    font-weight: 500;
}
.pred-pct {
    opacity: 0.7;
    font-variant-numeric: tabular-nums;
}
.pred-track {
    position: relative;
    height: 18px;
    background: var(--neutral-200, #e5e7eb);
    border-radius: 4px;
}
.pred-fill {
    height: 100%;
    background: var(--color-accent, #3b82f6);
    border-radius: 4px;
    transition: width 0.3s ease;
}
.pred-threshold {
    position: absolute;
    top: -2px;
    bottom: -2px;
    width: 2px;
    background: #ef4444;
    border-radius: 1px;
    z-index: 2;
}
.pred-threshold::before {
    content: '';
    position: absolute;
    top: -4px;
    left: -3px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 4px solid #ef4444;
}
.pred-legend {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 4px;
    font-size: 0.8em;
    opacity: 0.6;
}
.pred-legend-marker {
    display: inline-block;
    width: 2px;
    height: 14px;
    background: #ef4444;
    border-radius: 1px;
}
"""

lightbox_js = """
() => {
    document.addEventListener('click', (e) => {
        const img = e.target.closest('#image-input img');
        if (!img || !img.src) return;
        const overlay = document.createElement('div');
        overlay.className = 'image-lightbox';
        const enlarged = document.createElement('img');
        enlarged.src = img.src;
        overlay.appendChild(enlarged);
        overlay.onclick = () => overlay.remove();
        document.body.appendChild(overlay);
    });
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(title="IFCB Plankton Classifier") as demo:

    session = gr.State(None)
    page = gr.State(0)

    gr.HTML(
        "<h1 class='main-title'>IFCB Plankton Classifier</h1>"
        "<p class='subtitle'>"
        "Classify phytoplankton images using a fine-tuned ResNet-50"
        "</p>"
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload image",
                sources=["upload", "clipboard"],
                height=350,
                elem_id="image-input",
            )
            with gr.Row():
                classify_btn = gr.Button(
                    "Classify",
                    variant="primary",
                    size="lg",
                )
                zip_btn = gr.UploadButton(
                    "Upload ZIP",
                    file_types=[".zip"],
                    variant="secondary",
                    size="lg",
                )

        with gr.Column(scale=1):
            label_output = gr.HTML(
                value=render_predictions({}),
                label="Predictions",
            )

    with gr.Row(elem_id="sort-row"):
        gr.HTML("<span>Alphabetical</span>")
        sort_by_dim = gr.Checkbox(
            value=True,
            show_label=False,
            container=False,
            elem_id="sort-toggle",
        )
        gr.HTML("<span>Dimension</span>")

    gallery = gr.Gallery(
        label="Session images (click to classify)",
        columns=6,
        object_fit="contain",
        allow_preview=False,
    )

    with gr.Row():
        prev_btn = gr.Button("< Prev", size="sm", scale=1, interactive=False)
        with gr.Column(scale=2):
            page_info = gr.Markdown(
                "No images uploaded",
                elem_classes=["page-info"],
            )
        next_btn = gr.Button("Next >", size="sm", scale=1, interactive=False)

    classify_btn.click(
        fn=predict_html,
        inputs=[image_input],
        outputs=[label_output],
    )
    image_input.upload(
        fn=handle_image_upload,
        inputs=[image_input, session, page, sort_by_dim],
        outputs=[session, image_input, gallery, page, page_info, prev_btn, next_btn, label_output],
    )
    zip_btn.upload(
        fn=handle_zip_upload,
        inputs=[zip_btn, session, page, sort_by_dim],
        outputs=[session, gallery, page, page_info, prev_btn, next_btn],
    )
    gallery.select(
        fn=on_gallery_select,
        inputs=[session, page, sort_by_dim],
        outputs=[image_input, label_output],
    )
    prev_btn.click(
        fn=go_prev,
        inputs=[session, page, sort_by_dim],
        outputs=[gallery, page, page_info, prev_btn, next_btn],
    )
    next_btn.click(
        fn=go_next,
        inputs=[session, page, sort_by_dim],
        outputs=[gallery, page, page_info, prev_btn, next_btn],
    )
    sort_by_dim.change(
        fn=on_sort_change,
        inputs=[session, sort_by_dim],
        outputs=[gallery, page, page_info, prev_btn, next_btn],
    )

    with gr.Accordion("About", open=False):
        gr.Markdown(build_about_markdown())

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=theme,
    css=css,
    js=lightbox_js,
    max_file_size="50mb",
)
