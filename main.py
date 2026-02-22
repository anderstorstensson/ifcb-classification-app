import io
import os
import shutil
import zipfile

import gradio as gr
from PIL import Image

from model import (
    predict, predict_html, render_predictions,
    build_about_markdown, get_thresholds,
    AVAILABLE_MODELS, DEFAULT_MODEL,
)
from session import (
    init_session, save_image, sorted_indices,
    gallery_page, page_count, page_info_text,
    _active_session_dirs, _session_lock, log,
    IMAGE_EXTENSIONS, IMAGES_PER_PAGE,
    MAX_ZIP_BYTES, MAX_ZIP_FILES, MAX_SINGLE_FILE_BYTES,
)


# --- Event handlers ---


def disable_actions():
    return gr.Button(interactive=False), gr.UploadButton(interactive=False)


def enable_actions():
    return gr.Button(interactive=True), gr.UploadButton(interactive=True)


def capture_zip_and_disable(file):
    return file, gr.Button(interactive=False), gr.UploadButton(interactive=False)


def clear_session(session):
    if session and "dir" in session:
        shutil.rmtree(session["dir"], ignore_errors=True)
        with _session_lock:
            _active_session_dirs.discard(session["dir"])
    page = 0
    return (
        None, gr.Gallery(value=[], selected_index=None), page, "No images uploaded",
        gr.Image(value=None, label="Upload image"),
        render_predictions({}), gr.Textbox(value="", visible=False),
        gr.Button(interactive=False), gr.Button(interactive=False),
    )


def nav_buttons(session, page):
    pages = page_count(session)
    return (
        gr.Button(interactive=(page > 0)),
        gr.Button(interactive=(page < pages - 1)),
    )


async def handle_image_upload(image, session, page, sort_by_dim, model_name):
    if image is None:
        prev, nxt = nav_buttons(session, page)
        return session, gr.Image(label="Upload image"), gallery_page(session, page, sort_by_dim), page, page_info_text(session, page), prev, nxt, render_predictions({}), gr.Textbox(value="", visible=False)
    original_name = ""
    if hasattr(image, "filename") and image.filename:
        original_name = os.path.basename(image.filename)
    session = save_image(image, session, original_name)
    page = 0
    preds = await predict(image, model_name=model_name)
    prev, nxt = nav_buttons(session, page)
    label = os.path.splitext(original_name)[0] if original_name else "Upload image"
    return session, gr.Image(value=image, label=label), gallery_page(session, page, sort_by_dim), page, page_info_text(session, page), prev, nxt, render_predictions(preds, model_name=model_name), gr.Textbox(value=label if label != "Upload image" else "", visible=label != "Upload image")


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


def on_gallery_select(session, page, sort_by_dim, evt: gr.SelectData):
    indices = sorted_indices(session, sort_by_dim)
    pos = page * IMAGES_PER_PAGE + evt.index
    if pos < 0 or pos >= len(indices):
        return gr.Image(value=None, label="Upload image"), gr.Textbox(value="", visible=False)
    idx = indices[pos]
    paths = session["paths"]
    names = session.get("names", [])
    img = Image.open(paths[idx])
    name = names[idx] if idx < len(names) else ""
    label = os.path.splitext(name)[0] if name else "Upload image"
    return gr.Image(value=img, label=label), gr.Textbox(value=label if label != "Upload image" else "", visible=label != "Upload image")


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


def update_about(model_name):
    return build_about_markdown(model_name)


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
    zip_file = gr.State(None)

    gr.HTML(
        "<h1 class='main-title'>IFCB Plankton Classifier</h1>"
        "<p class='subtitle'>"
        "Classify phytoplankton images using a fine-tuned ResNet-50 for the Skagerrak, Kattegat, and Baltic sea"
        "</p>"
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                format="png",
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
                clear_btn = gr.Button(
                    "Clear",
                    variant="stop",
                    size="lg",
                )
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=DEFAULT_MODEL,
                label="Model",
            )

        with gr.Column(scale=1):
            label_output = gr.HTML(
                value=render_predictions({}),
                label="Predictions",
            )
            filename_box = gr.Textbox(
                label="Filename",
                interactive=False,
                buttons=["copy"],
                visible=False,
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
        fn=disable_actions,
        outputs=[classify_btn, zip_btn],
    ).then(
        fn=predict_html,
        inputs=[image_input, model_dropdown],
        outputs=[label_output],
    ).then(
        fn=enable_actions,
        outputs=[classify_btn, zip_btn],
    )
    image_input.upload(
        fn=disable_actions,
        outputs=[classify_btn, zip_btn],
    ).then(
        fn=handle_image_upload,
        inputs=[image_input, session, page, sort_by_dim, model_dropdown],
        outputs=[session, image_input, gallery, page, page_info, prev_btn, next_btn, label_output, filename_box],
    ).then(
        fn=enable_actions,
        outputs=[classify_btn, zip_btn],
    )
    zip_btn.upload(
        fn=capture_zip_and_disable,
        inputs=[zip_btn],
        outputs=[zip_file, classify_btn, zip_btn],
    ).then(
        fn=handle_zip_upload,
        inputs=[zip_file, session, page, sort_by_dim],
        outputs=[session, gallery, page, page_info, prev_btn, next_btn],
    ).then(
        fn=enable_actions,
        outputs=[classify_btn, zip_btn],
    )
    clear_btn.click(
        fn=clear_session,
        inputs=[session],
        outputs=[session, gallery, page, page_info, image_input, label_output, filename_box, prev_btn, next_btn],
    )
    gallery.select(
        fn=on_gallery_select,
        inputs=[session, page, sort_by_dim],
        outputs=[image_input, filename_box],
    ).then(
        fn=predict_html,
        inputs=[image_input, model_dropdown],
        outputs=[label_output],
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
        about_md = gr.Markdown(build_about_markdown(DEFAULT_MODEL))

    model_dropdown.change(
        fn=update_about,
        inputs=[model_dropdown],
        outputs=[about_md],
    )

    # --- API-only endpoint (not visible in the UI) ---
    thresholds_btn = gr.Button(visible=False)
    thresholds_model = gr.Dropdown(
        choices=list(AVAILABLE_MODELS.keys()),
        value=DEFAULT_MODEL,
        visible=False,
    )
    thresholds_output = gr.JSON(visible=False)
    thresholds_btn.click(
        fn=get_thresholds,
        inputs=[thresholds_model],
        outputs=[thresholds_output],
        api_name="get_thresholds",
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=theme,
    css=css,
    js=lightbox_js,
    max_file_size="50mb",
)
