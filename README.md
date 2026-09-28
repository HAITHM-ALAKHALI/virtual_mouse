<p align="center">
  <h1 align="center">A Hybrid Architecture for Vision-Based Virtual Mouse Systems</h1>
  <p align="center">
    <em>Normalized Gesture Classification and Deterministic OS Event Dispatching</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT">
  <img src="https://img.shields.io/badge/MediaPipe-Tasks_API-00A98F?logo=google&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Dataset-Kaggle_v1.0.0-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=white" alt="arXiv">
</p>

---

## Abstract

This repository contains the reference implementation for the paper:

> **A Hybrid Architecture for Vision-Based Virtual Mouse Systems: Normalized Gesture Classification and Deterministic OS Event Dispatching**
> *Haithm Abdulwali Al-Akhali — Department of Computer Science, Sana'a University*

The system presents an empirically evaluated hybrid architecture for real-time virtual mouse control that bridges probabilistic computer vision inference with deterministic operating system event dispatching. The pipeline decouples spatial landmark kinematics from intentional state recognition: 21 hand landmarks are extracted via MediaPipe Tasks (`RunningMode.VIDEO`), from which 2D image-plane components are retained and normalised into a **42-dimensional feature vector** — translation-normalised to the wrist origin, scale-normalised against the wrist-to-middle-MCP distance, and canonicalised across handedness via horizontal reflection.

An **RBF-kernel SVM pipeline** (C=10.0, StandardScaler, 1,139 support vectors) trained on **10,008 multi-subject instances** across six gesture classes achieves **98.10% test accuracy** with **0.717 ms** isolated inference latency. Downstream, a **deterministic action dispatcher** enforces time-gated debouncing (0.25 s), coordinate freezing during click actuation, a latched context-menu state machine, and multi-frame drag hysteresis — all decoupled from frame rate. A **thread-safe circular telemetry buffer** (`deque(maxlen=20)`) provides millisecond-resolution audit tracing without blocking the vision processing loop.

---

## Key Architectural Features

- **42D Anatomical Normalisation Pipeline**
  - Translation normalisation: wrist landmark (p₀) as coordinate origin
  - Scale normalisation: Euclidean distance from wrist to middle-MCP joint (p₉ − p₀) as the scale divisor
  - Handedness canonicalisation: horizontal reflection (x ← −x) for left-hand inputs

- **Deterministic Action Dispatcher**
  - FPS-decoupled click debouncing: `T_interval = max(0.05, clickDelay × 0.025) s` (default 0.25 s), evaluated via `time.time()`
  - Coordinate freezing: OS pointer held at `(plocX, plocY)` during click actuation to prevent drag-slip
  - Context-menu latch (`_right_clicked`): suppresses redundant right-click dispatches; resets only on gesture transition or hand loss
  - Drag hysteresis: 2-frame persistence window (`drag_change_frames >= 2`) before releasing `Button.left`

- **Temporal Anchor Tracking**
  - Wrist-based hand identity tracking with D_max = 0.25 (normalised image coordinates)
  - Lost-frame hysteresis retaining anchor for up to M_lost = 20 consecutive frames

- **Dual-Thread Asynchronous Telemetry**
  - Vision + dispatch loop in an isolated daemon thread
  - Mutex-protected circular buffer (`_log_lock` + `deque(maxlen=20)`)
  - Millisecond-resolution audit traces: `[HH:MM:SS.mmm] ACTION | Pos: (X, Y) | Metadata`

---

## Repository Structure

```
├── Agent_Execution/
│   ├── App_GUI.py                  # Dark-themed control panel (CustomTkinter)
│   └── Main.py                     # VirtualMouseController — daemon thread pipeline
│
├── Data_Engineering/
│   ├── Dataset/
│   │   ├── Collect_Data.py         # Live landmark data collection with key-labelling
│   │   └── virtual_mouse_data.csv  # 10,008 labelled instances (42 features + label)
│   └── data_cleaning.ipynb         # Exploratory analysis and preprocessing
│
├── Model_Training/
│   ├── Train_Model.ipynb           # Classifier benchmarking (RF, SVM-RBF, MLP)
│   └── virtual_mouse_brain.pkl     # Deployed SVM pipeline (StandardScaler + SVC)
│
├── Test/
│   └── benchmark_latency.py        # Per-stage pipeline latency profiler (500 frames)
│
├── Utils/
│   ├── HandTrackingModule.py       # MediaPipe HandLandmarker wrapper + anchor tracker
│   └── hand_landmarker.task        # MediaPipe model asset
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

| Requirement | Version |
|:------------|:--------|
| Python | 3.10+ |
| OS | Windows 10/11 (primary), macOS / Linux (experimental) |
| Hardware | Webcam, 4+ CPU cores recommended |

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/HAITHM-ALAKHALI/virtual_mouse.git
cd virtual_mouse

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|:--------|:--------|
| `mediapipe` | Hand landmark detection (Tasks API, `RunningMode.VIDEO`) |
| `opencv-python` | Camera capture and image processing |
| `scikit-learn` | SVM classifier pipeline (StandardScaler + SVC) |
| `pynput` | OS-level mouse event injection (`Controller`, `Button`) |
| `customtkinter` | Modern dark-themed GUI framework |
| `Pillow` | BGR → RGB image conversion for GUI preview |
| `numpy` | Array operations and interpolation |
| `joblib` | Model serialisation (benchmark script) |

---

## Quick Start

### 1. Run the Live Application

```bash
cd Agent_Execution
python App_GUI.py
```

The GUI launches → click **▶ Start Virtual Mouse** → show your hand to the camera.

### 2. Collect Custom Training Data

```bash
cd Data_Engineering/Dataset
python Collect_Data.py
```

Hold a gesture and press the corresponding key to label frames:

| Key | Gesture |
|:---:|:--------|
| `m` | Move |
| `l` | Left_Click |
| `r` | Right_Click |
| `s` | Scroll |
| `d` | Drag |
| `n` | Neutral |
| `q` | Quit |

Labelled 42D feature vectors are appended to `virtual_mouse_data.csv`.

### 3. Train and Evaluate the ML Pipeline

Open and run `Model_Training/Train_Model.ipynb`. The notebook:
1. Loads `virtual_mouse_data.csv` (10,008 instances, 6 classes)
2. Applies an 80/20 stratified split (`test_size=0.20`, `random_state=42`, `stratify=y`)
3. Benchmarks Random Forest, SVM-RBF, and MLP pipelines
4. Exports the selected model to `virtual_mouse_brain.pkl`

### 4. Run the Latency Benchmark

```bash
cd Test
python benchmark_latency.py
```

Profiles 500 consecutive frames across 4 stages: MediaPipe detection, 42D normalisation, SVM inference, and action dispatch overhead.

### 5. Inspect Audit Traces

Use the **📥 Export Log** button in the GUI to save the telemetry audit trace. The exported file follows the format:

```
[HH:MM:SS.mmm] LEFT_CLICK   | Pos: (774, 533) | Debounce: 0.25s
[HH:MM:SS.mmm] RIGHT_CLICK  | Pos: (803, 726) | Latched
[HH:MM:SS.mmm] DRAG_START   | Anchor: (791, 753)
[HH:MM:SS.mmm] DRAG_DROP    | Released at: (493, 241)
[HH:MM:SS.mmm] SCROLL_UP    | Steps: 1 | ΔY: 85
[HH:MM:SS.mmm] SCROLL_DOWN  | Steps: 1 | ΔY: 227
```

---

## Dataset & Pre-trained Weights

| Asset | Location | Description |
|:------|:---------|:------------|
| Training dataset | [`Data_Engineering/Dataset/virtual_mouse_data.csv`](Data_Engineering/Dataset/virtual_mouse_data.csv) | 10,008 instances × 42 features + label |
| Kaggle dataset | [kaggle.com/datasets/haithmalakhali/virtual-mouse-landmarks](https://www.kaggle.com/datasets/haithmalakhali/virtual-mouse-landmarks) | Release v1.0.0 |
| Trained model | [`Model_Training/virtual_mouse_brain.pkl`](Model_Training/virtual_mouse_brain.pkl) | `Pipeline([StandardScaler, SVC(rbf, C=10.0)])` — 1,139 SVs |

Place `virtual_mouse_brain.pkl` in `Model_Training/` (default location) or alongside `Main.py` in `Agent_Execution/`.

---

## Gesture Reference

| Gesture | Hand Pose | OS Action |
|:--------|:----------|:----------|
| ☝️ **Move** | Index finger extended | Cursor follows index fingertip |
| ✌️ **Left Click** | Index + Middle fingers | Time-gated left click (0.25 s debounce) |
| 🤟 **Right Click** | Index + Middle + Ring | Latched right click (single dispatch) |
| 🖐️ **Scroll** | Open palm | Vertical scroll based on hand position relative to deadzone |
| ✊ **Drag** | Closed fist | Press-hold `Button.left`; release after 2-frame hysteresis |

---

## Runtime Configuration

All settings are adjustable in real time from the GUI:

| Setting | Range | Default | Description |
|:--------|:-----:|:-------:|:------------|
| **Smoothening** (α) | 1 – 15 | 7 | Exponential moving average divisor for cursor position |
| **Scroll Deadzone** | 20 – 150 px | 60 | Vertical dead band around camera centre; suppresses unintended scrolling |
| **Scroll Speed** | 1 – 10 | 3 | Controls scroll step size and inter-scroll frame delay |
| **Click Delay** | 0 – 20 | 10 | Maps to `T_interval = max(0.05, value × 0.025) s` |

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{alakhali2026hybrid,
  title   = {A Hybrid Architecture for Vision-Based Virtual Mouse Systems:
             Normalized Gesture Classification and Deterministic OS Event
             Dispatching},
  author  = {Al-Akhali, Haithm Abdulwali},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Author

**Haithm Abdulwali Al-Akhali**
Department of Computer Science, Faculty of Computer Science and Information Technology
Sana'a University, Sana'a, Yemen

[🌐 Portfolio](https://haithm-al-akhali.vercel.app/) · [💻 GitHub](https://github.com/HAITHM-ALAKHALI) · [✉️ Email](mailto:haithmabdulwali.alakhali@gmail.com)