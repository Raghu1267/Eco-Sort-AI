# ECO SORT AI

**AI-Based Smart E-Waste Segregation and Classification System Using YOLOv8**

A Streamlit web app that detects e-waste in uploaded images with a trained YOLOv8s model (`model/best.pt`, 37 classes) and then uses a rule-based engine to suggest a category, collection stream, hazard priority, disposal method, recycling route and safety precautions for each detected item.

- Image upload only (JPG, JPEG, PNG). No webcam, hardware or sensors.
- CPU inference. No Google Colab or Google Drive needed.
- Every detection comes from real inference with your `best.pt`. Nothing is simulated.

> The segregation output is a general software recommendation. It is **not** a legally certified waste classification. Always follow your local authority's disposal rules.

## Project structure

```
eco_sort_ai/
├── app.py                  # Streamlit app: 5 pages (Home, AI Detection, Smart Segregation, Analytics, About)
├── model/best.pt           # Trained YOLOv8s weights (37 classes)
├── modules/
│   ├── detection.py        # Model loading, image validation, inference, box drawing
│   ├── segregation.py      # Rule-based segregation engine for the 37 classes
│   ├── charts.py           # Plotly charts for the analytics page
│   └── ui.py               # Theme CSS and HTML components
├── assets/logo.svg
├── .streamlit/config.toml  # Dark navy/teal theme and upload limit
├── verify_setup.py         # Self-check: model loads, class names match, CPU inference runs
├── requirements.txt        # Python dependencies (CPU-only PyTorch)
├── packages.txt            # System libraries for Streamlit Community Cloud (OpenCV)
├── README.md
└── .gitignore
```

## Install and run locally

Requirements: Python 3.10 to 3.12 (3.11 or 3.12 recommended) and about 2 GB of free disk space.

```bash
cd eco_sort_ai

# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Recommended) check the model and the class mapping
python verify_setup.py
python verify_setup.py path/to/some_photo.jpg   # optional: test a real image

# 4. Start the app
streamlit run app.py
```

The app opens at http://localhost:8501.

### What `verify_setup.py` checks

1. `model/best.pt` loads.
2. The model has 37 classes and each one has a segregation rule (names must match exactly).
3. A real CPU inference runs without errors.

## Upload to GitHub

1. Create an empty repository on GitHub (for example `eco-sort-ai`). Do not add a README there.
2. In the project folder run:

```bash
git init
git add .
git commit -m "ECO SORT AI: YOLOv8 e-waste segregation app"
git branch -M main
git remote add origin https://github.com/<your-username>/eco-sort-ai.git
git push -u origin main
```

`model/best.pt` is about 22 MB, which is under GitHub's 100 MB file limit, so Git LFS is not needed. Do not add your training dataset to the repository.

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with your GitHub account.
2. Click **Create app** and pick your repository and the `main` branch.
3. Set **Main file path** to `app.py`.
4. Open **Advanced settings** and choose **Python 3.11** or **3.12**.
5. Click **Deploy**. The first build installs PyTorch and Ultralytics and takes several minutes.

`requirements.txt` points to the CPU-only PyTorch wheel index and `packages.txt` installs the system libraries OpenCV needs, so no extra setup is required.

## How it works

1. The upload is validated (type, size up to 15 MB, real JPEG or PNG content) and corrected for phone rotation. Very large images are scaled to 3000 px on the long side.
2. The model is loaded once with `st.cache_resource` and run on CPU at 640 px.
3. Boxes, class names (read from the model) and confidences are extracted and drawn.
4. Each detected class is looked up in `modules/segregation.py`.
5. Results are shown in tables and charts and can be downloaded as a PNG and as CSV files.

## Model performance

Results from the test-set evaluation of the trained YOLOv8s model:

| Metric | Value |
|---|---|
| Precision | 92.89% |
| Recall | 86.33% |
| F1-score | 89.49% |
| mAP@50 | 92.04% |
| mAP@50-95 | 75.94% |

mAP is an object-detection metric. It is not classification accuracy.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` | Activate your virtual environment and run `pip install -r requirements.txt`. |
| `Model file not found` | Make sure the weights are at `model/best.pt` and you start the app from the project folder or with `streamlit run path/to/app.py`. |
| Model fails to load with an Ultralytics or `weights_only` error | The weights were saved with Ultralytics 8.4.216. Run `pip install -U ultralytics` (or install that exact version) and try again. |
| `ImportError: libGL.so.1` | Locally: `sudo apt-get install libgl1 libglib2.0-0`. On Streamlit Cloud, keep `packages.txt` in the repository root. |
| Streamlit Cloud build runs out of memory or space | Confirm `requirements.txt` still starts with the `--extra-index-url .../whl/cpu` line so the small CPU build of PyTorch is used. |
| Detection is slow | CPU inference takes a second or more per image. The first run is slower because the model is loading. |
| Nothing is detected | Lower the confidence threshold, use a sharper photo and make the item fill more of the frame. The model only knows its 37 classes. |
| Upload rejected | Use a real JPG, JPEG or PNG under 15 MB. A renamed file of another type is rejected. |
| Port already in use | `streamlit run app.py --server.port 8502` |

## Licence note

Ultralytics YOLOv8 is released under AGPL-3.0. If you publish this app, keep the source available (a public GitHub repository does this) and check the Ultralytics licence terms for your use.
