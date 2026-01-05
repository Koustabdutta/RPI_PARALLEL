# Parallel YOLOv8 + MiDaS GUI (Accuracy-first)

A small desktop GUI for parallel (YOLOv8 + MiDaS) ONNX inference with letterbox preprocessing, robust ONNX output handling, NMS, confidence/IoU sliders and a simple system monitor.
The main app file is `main_gui_4.py`. 

---

## Features

* Run YOLOv8 object detection and MiDaS depth estimation in parallel threads.
* Robust parsing of different ONNX output layouts (handles `(1,C,N)` and `(1,N,C)`).
* Confidence threshold and NMS IoU sliders.
* Model selector dropdown (pick `yolov8s.onnx` / `yolov8m.onnx` / `yolov8n.onnx` etc.).
* Simple system CPU usage display and per-frame timing metrics.
* Combined overlay of detections + depth map.

---

## Repository contents

* `main_gui_4.py` — main GUI application (run this). 
* `export_onnx.ipynb` — (optional) notebook used to export/prepare ONNX models (if included).
* `README.md` — (this file)

---

## Requirements

* Python 3.8+
* The following Python packages:

```
pip install opencv-python numpy onnxruntime customtkinter pillow psutil
```

(If you prefer, create a venv / conda env first.)

**Tip:** ONNX models can be large — consider using Git LFS for `.onnx` files:

```
git lfs install
git lfs track "*.onnx"
```

---

## Required model files (place in project root)

* `yolov8n.onnx` / `yolov8s.onnx` / `yolov8m.onnx` (any `yolov8*.onnx` — the UI will list found files).
* `Midas-V2.onnx` — MiDaS depth model (expected filename is `Midas-V2.onnx`).

If a required model is missing, the GUI will warn and disable the Run button.

---

## Quick start

1. Clone / create repository and add files:

```bash
git init
git add main_gui_4.py export_onnx.ipynb README.md
git commit -m "Initial commit: YOLOv8 + MiDaS GUI"
# (optionally set remote and push)
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. Make sure models are in the repo folder (or copy them into the same folder). If large, store them elsewhere and document download links in your repo.

3. Run the GUI:

```bash
python3 main_gui_4.py
```

* Click **Load Image**, select an image, choose an ONNX model from the dropdown, adjust confidence / IoU sliders, then **Run Parallel Inference**.

---

## Usage notes & troubleshooting

* If no ONNX models are detected, the dropdown will still show a default name but the run button will be disabled — copy a valid `yolov8*.onnx` and `Midas-V2.onnx` into the project folder.
* The GUI expects RGB images internally. If `cv2.imread()` fails to read an image, ensure the file is a standard JPG/PNG.
* If you get model-loading errors, ensure `onnxruntime` is installed and compatible with your platform (CPU providers are used by default).
* For GPU acceleration, replace the `providers` argument when creating ONNX sessions (requires GPU-capable `onnxruntime` builds) — be mindful of platform compatibility.

---

## Development / Extending

* Improve class list: the current `COCO_CLASSES` is embedded in the script — update if using a different dataset.
* Add ONNX GPU provider config for faster inference.
* Add a video/webcam mode (frame loop + async queue).
* Convert the UI to allow relative/absolute paths for models and images.

---

## Attribution

This project is a local GUI wrapper around YOLOv8 ONNX exports and MiDaS ONNX — inference and preprocessing code are implemented in `main_gui_4.py`.

Copy-paste this entire file as `README.md` into your repository. Need a `requirements.txt`, `.gitignore`, or a short repo description blurb for GitHub? I can generate them immediately.
