# IFCB Classification App

A web application for classifying phytoplankton images from Imaging FlowCytobot (IFCB) instruments using a fine-tuned ResNet-50 model for the Skagerrak, Kattegat, and Baltic sea. Built with [Gradio](https://www.gradio.app/).

## Features

- Single image classification with top-5 predictions and confidence bars
- Batch classification via ZIP upload (up to 10,000 images)
- Paginated image gallery with sorting by name or dimension
- Per-class F2 optimised thresholds displayed on prediction bars
- Auto-discovery of models from the `data/models/` directory
- API endpoints for programmatic access (`predict_scores`, `get_thresholds`)

## Models

The app auto-discovers models from the `data/models/` directory. Each model needs a subdirectory containing `weights.pth`, `classes.txt`, and optionally `thresholds.json` and `about.md`. Model weights are not included in the repository — contact the author to obtain them.

The included example configuration (SMHI-NIVA-ResNet50-V5) expects:

- **Architecture:** ResNet-50
- **Input:** 224 x 224 px (square-padded with adaptive background colour)
- **Classes:** 109
- **Training data:** [SMHI IFCB Plankton Image Reference Library](https://doi.org/10.17044/scilifelab.25883455) and images provided by the Norwegian Institute for Water Research (NIVA)

## Requirements

- Python 3.11+
- PyTorch (CPU-only by default via `requirements.txt`)

### GPU Support

The default `requirements.txt` installs CPU-only PyTorch. The app automatically detects and uses CUDA or MPS (Apple Silicon) when available. To enable GPU acceleration, reinstall PyTorch with CUDA support:

```bash
pip install -r requirements.txt

# Then reinstall PyTorch with CUDA (check https://pytorch.org/get-started for the latest commands)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Local Setup

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat

pip install -r requirements.txt
python main.py
```

The app starts at `http://localhost:7860`.

## Docker

```bash
docker build -t ifcb-classification-app .
docker run -p 7860:7860 ifcb-classification-app
```

## API Endpoints

The app exposes two API endpoints alongside the web UI:

- **`predict_scores`** — Classify an image and return all class scores as JSON (`{"class_labels": [...], "scores": [...]}`).
- **`get_thresholds`** — Return per-class F2 thresholds and class labels for a model.

See the Gradio API docs at `http://localhost:7860/?view=api` when the app is running.

## Project Structure

```
├── main.py                 # Event handlers, UI layout, app entry point
├── model.py                # Model loading, inference, prediction rendering
├── session.py              # Session state, ZIP handling, gallery helpers
├── utils/
│   └── CustomTransforms.py # Square-pad transform
├── data/
│   └── models/
│       └── SMHI-NIVA-ResNet50-V5/
│           ├── weights.pth      # Model weights (git-ignored)
│           ├── classes.txt      # Class labels
│           ├── thresholds.json  # Per-class F2 thresholds
│           └── about.md         # Model description (optional)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata and version
├── Dockerfile
└── LICENSE
```

## License

[MIT](LICENSE)
