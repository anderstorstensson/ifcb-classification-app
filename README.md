# IFCB Classification App

A web application for classifying phytoplankton images from Imaging FlowCytobot (IFCB) instruments using a fine-tuned ResNet-50 model for the Skagerrak, Kattegat, and Baltic sea. Built with [Gradio](https://www.gradio.app/).

## Features

- Single image classification with top-5 predictions and confidence bars
- Batch classification via ZIP upload (up to 10,000 images)
- Paginated image gallery with sorting by name or dimension
- Per-class F2 optimised thresholds displayed on prediction bars
- Lightbox view for uploaded images

## Model

- **Architecture:** ResNet-50
- **Input:** 224 x 224 px (square-padded with adaptive background colour)
- **Classes:** 108
- **Training data:** [SMHI IFCB Plankton Image Reference Library](https://doi.org/10.17044/scilifelab.25883455) and images provided by the Norwegian Institute for Water Research (NIVA)

## Requirements

- Python 3.11+
- Model weights (`weights.pth`) in `data/models/ifcb-plankton-resnet50/` — not included in the repository, contact the author to obtain them

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

## Project Structure

```
├── main.py                 # Application entry point and UI
├── utils/
│   └── CustomTransforms.py # Square-pad transform
├── data/
│   └── models/
│       └── ifcb-plankton-resnet50/
│           ├── weights.pth      # Model weights (git-ignored)
│           ├── classes.txt      # Class labels
│           └── thresholds.json  # Per-class F2 thresholds
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata and version
├── Dockerfile
└── LICENSE
```

## License

[MIT](LICENSE)
