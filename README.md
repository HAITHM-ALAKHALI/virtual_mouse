<div align="center">

# A Hybrid Architecture for Vision-Based Virtual Mouse Systems
*Normalized Gesture Classification and Deterministic OS Event Dispatching*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-00A98F?style=flat-square&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle_v1.0.0-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/haithmalakhali/virtual-mouse-landmarks)
[![IEEE Paper](https://img.shields.io/badge/Paper-IEEE_Format-00629B?style=flat-square&logo=ieee&logoColor=white)](https://github.com/HAITHM-ALAKHALI/virtual_mouse)

</div>

---

## Abstract

This repository contains the official reference implementation, benchmark scripts, and empirical telemetry logs for the paper:

> **A Hybrid Architecture for Vision-Based Virtual Mouse Systems: Normalized Gesture Classification and Deterministic OS Event Dispatching**  
> *Haithm Abdulwali Al-Akhali — Department of Computer Science, Faculty of Computer Science and Information Technology, Sana'a University*

Vision-based human-computer interaction (HCI) interfaces frequently suffer from operational non-determinism, including high-frequency cursor jitter, unintentional multi-click cascades, and unintended context-menu dismissals caused by hardware-dependent frame rate (FPS) variations and involuntary physiological hand tremors. This project presents an empirically evaluated hybrid architecture for real-time virtual mouse control that bridges probabilistic computer vision inference with rule-based operating system (OS) event-dispatching constraints.

The pipeline decouples spatial landmark kinematics from intentional state recognition: 21 three-dimensional hand landmarks are extracted via MediaPipe Tasks (`RunningMode.VIDEO`), stabilized using a temporal wrist-anchoring state machine to prevent multi-user target switching. Retained 2D image-plane components undergo translation normalization to the wrist origin, approximate uniform-scale normalization against the wrist-to-middle-MCP distance, and handedness canonicalization via horizontal reflection into a unified **42-dimensional feature vector**.

Evaluated on a curated dataset of **10,008 empirical instances** across five human subjects and six operational classes (`Move`, `Left_Click`, `Right_Click`, `Drag`, `Scroll`, and `Neutral`), an **RBF-kernel Support Vector Machine (SVM)** pipeline achieves **98.10% sample-level test accuracy** under a stratified random split with an isolated inference latency of **0.717 ms** (averaging **0.72 ms** under steady-state daemon execution) and rapid model training time (**0.68 s**). Crucially, zero-shot evaluation of the frozen classifier on an external held-out cohort of **766 newly captured samples** across three unseen human subjects (`p1.csv`, `p2.csv`, `p3.csv`) demonstrates **88.12% cross-subject accuracy** (macro F1-score: **0.88**) without subject-specific retraining. Downstream, an FPS-decoupled, wall-clock debouncing mechanism (0.25 s) combined with coordinate freezing, an event-latched state machine, and two-frame drag hysteresis suppresses multi-click cascades, retains context menus, and maintains stable logged actuation coordinates across exploratory live ablation trials, with an average post-detection pipeline overhead of approximately **1.22 ms**.

---

## Key Architectural Features

- **Temporal Target-Hand Anchoring (`Utils/HandTrackingModule.py`)**
  - Wrist-based hand identity tracking with Euclidean distance threshold `D_max = 0.25` in normalized image space.
  - Tracking-loss hysteresis counter retaining anchor parameters for up to `M_lost = 20` consecutive frames to prevent disruptive target switching.

- **Bifurcated 42D Anatomical Normalization Pipeline**
  - **Continuous Kinematic Stream:** Maps raw index fingertip `p_8 = (x_8, y_8)` to display resolution `(w_scr, h_scr)` with exponential smoothing divisor `alpha` in `[1, 15]`.
  - **Normalized Classification Stream:**
    1. *Translation Normalization:* Wrist landmark (`p_0`) set as the coordinate origin (`q_i = p_i - p_0`).
    2. *Anatomical Scale Normalization:* Scaled by characteristic palm length `s = ||p_9 - p_0||_2` with an explicit division-by-zero safeguard.
    3. *Handedness Canonicalization:* Horizontal reflection (`x_i <- -x_i` if `Handedness == Left`) to unify left and right hands into a single feature space.

- **Rule-Based Action Dispatcher (`Agent_Execution/Main.py`)**
  - **FPS-Decoupled Wall-Clock Debouncing:** Enforces `Delta_t = t_now - t_last_click >= T_interval` (`0.25 s` default) via `time.time()`.
  - **Actuation Coordinate Freezing:** Locks OS pointer at pre-actuation anchor `(plocX, plocY)` during `Left_Click` and `Right_Click` actuation to mitigate finger-flexion slip.
  - **Latched Context-Menu State Machine (`L_R`):** Dispatches a single right-click event and latches state, suppressing redundant frame triggers until explicit gesture transition or hand loss.
  - **Two-Frame Drag Hysteresis (`Delta_f >= 2`):** Requires two consecutive non-drag frames before releasing `Button.left`, absorbing transient single-frame classification flickers.
  - **Scroll Deadzone Engine:** Evaluates vertical displacement relative to a configurable center deadzone (`|y_8 - Y_center| > deadzone`) with dynamic velocity scaling.

- **Dual-Thread Asynchronous Telemetry & Full Session Auditing**
  - Vision acquisition, inference, and OS dispatching execute in an isolated daemon thread decoupled from the `CustomTkinter` GUI main loop.
  - Protected by synchronization mutexes (`_frame_lock` and `_log_lock`) with a live circular preview buffer (`deque(maxlen=20)`) and full-session export buffers logging both dispatched actions and `SUPPRESSED` cooldown candidates at millisecond resolution.

---

## Experimental Benchmarks & Results

### 1. Classifier Pipeline Comparison (2,002 Test Instances — Stratified 80/20 Split)

| Classifier Architecture | Accuracy (%) | Macro Precision | Macro Recall | Macro F1-Score | Isolated Latency (ms) | Training Time (s) |
|:------------------------|:------------:|:---------------:|:------------:|:--------------:|:---------------------:|:-----------------:|
| Random Forest (100 Estimators) | 97.35% | 0.97 | 0.97 | 0.97 | 9.117 | 4.43 |
| **SVM (RBF Kernel, C=10.0)** | **98.10%** | **0.98** | **0.98** | **0.98** | **0.717** | **0.68** |
| Neural Network (MLP, 64x32) | 98.20% | 0.98 | 0.98 | 0.98 | 0.591 | 14.66 |

### 2. Zero-Shot Cross-Subject Generalization (Unseen External Cohort)

Evaluated directly using the frozen `virtual_mouse_brain.pkl` model on `p1.csv`, `p2.csv`, and `p3.csv` without any subject-specific fine-tuning or calibration:

| Evaluation Cohort | Subjects | Dataset File | Sample Count | Accuracy (%) | Macro F1-Score |
|:------------------|:--------:|:-------------|:------------:|:------------:|:--------------:|
| Internal Test Split | 5 (Shared) | `virtual_mouse_data.csv` | 2,002 | 98.10% | 0.98 |
| Subject 1 (External) | 1 (Unseen) | `p1.csv` | 292 | 81.51% | — |
| Subject 2 (External) | 1 (Unseen) | `p2.csv` | 212 | 100.00% | — |
| Subject 3 (External) | 1 (Unseen) | `p3.csv` | 262 | 85.88% | — |
| **Combined Held-Out Cohort** | **3 (Unseen)** | `p1.csv` – `p3.csv` | **766** | **88.12%** | **0.88** |

### 3. Exploratory Live Ablation Study of Action Dispatcher Mechanisms

Conducted across 6 independent live interactive camera sessions isolating each software control in `Main.py` and auditing raw internal telemetry traces located in `Test/`:

| Configuration Pipeline | Audit Log File (`Test/`) | Logged Drift Span (px) | Cascade Log Events | Redundant Menu Events | Premature Drops (Drops / Starts) |
|:-----------------------|:-------------------------|:----------------------:|:------------------:|:---------------------:|:--------------------------------:|
| *Naive Baseline (All Off)* | [`audit_baseline.txt`](Test/audit_baseline.txt) | 99.92 | 59 | 48 | 1 / 1 (100.0%) |
| w/o Actuation Coord Freeze | [`audit_wo_freeze.txt`](Test/audit_wo_freeze.txt) | 54.41 | 0 | 0 | 0 / 0 (0.0%) |
| w/o Wall-Clock Debounce (Delta_t) | [`audit_wo_debounce.txt`](Test/audit_wo_debounce.txt) | **0.00** | 72 (71 excess) | 0 | 0 / 0 (0.0%) |
| w/o Latched Menu FSM (L_R) | [`audit_wo_latch.txt`](Test/audit_wo_latch.txt) | **0.00** | 0 | 61 (60 seq.) | 0 / 0 (0.0%) |
| w/o Drag Hysteresis (Delta_f >= 2) | [`audit_wo_drag.txt`](Test/audit_wo_drag.txt) | **0.00** | 0 | 0 | 7 / 10 (70.0%) |
| **Full Proposed Pipeline** | [`audit_proposed.txt`](Test/audit_proposed.txt) | **0.00** | **0** | **0** | **0 / 1 (0.0%)** |

> **Methodological Note:** Because each ablation condition was evaluated in a separate live webcam session, the recorded event counts and coordinate spans represent descriptive telemetry observations under the tested conditions rather than trial-matched causal estimates from identical video replays. In `audit_proposed.txt`, the full configuration logged 9 accepted `LEFT_CLICK` events (38 cooldown candidates suppressed), 2 accepted `RIGHT_CLICK` events (75 held-state candidates suppressed), 1 `DRAG_START`, 1 completed `DRAG_DROP` (0 premature drops), and 8 `SCROLL_UP` events (10 deadzone candidates suppressed).

### 4. End-to-End Runtime Latency Profile (500 Operational Cycles)

| Pipeline Stage | Mean Latency ± Std (ms) | Notes |
|:---------------|:-----------------------:|:------|
| 1. Video Acquisition & MediaPipe Inference | 33.53 ± 38.54 ms | 95th percentile: 62.93 ms |
| 2. 42D Anatomical Normalization | 0.06 ± 0.01 ms | Translation, scale, handedness reflection |
| 3. SVM-RBF Classification | 0.72 ± 0.14 ms | 0.717 ms isolated micro-benchmark |
| 4. Action Dispatcher & OS Event Injection | 0.44 ± 0.08 ms | Includes mutex telemetry logging + `pynput` |
| **Total Post-Detection Overhead (Stages 2–4)** | **~1.22 ms** | **< 4% of nominal 30 FPS frame budget** |

---

## Repository Structure

~~~text
├── Agent_Execution/
│   ├── App_GUI.py                  # Dark-themed control panel & full-session log exporter (CustomTkinter)
│   └── Main.py                     # VirtualMouseController — daemon thread pipeline & ablation flags
│
├── Data_Engineering/
│   ├── Dataset/
│   │   ├── Collect_Data.py         # Live landmark data collection with key-labelling
│   │   ├── virtual_mouse_data.csv  # 10,008 labelled development instances (5 subjects)
│   │   ├── p1.csv                  # Unseen held-out Subject 1 dataset (292 instances)
│   │   ├── p2.csv                  # Unseen held-out Subject 2 dataset (212 instances)
│   │   └── p3.csv                  # Unseen held-out Subject 3 dataset (262 instances)
│   └── data_cleaning.ipynb         # Exploratory analysis and preprocessing
│
├── Model_Training/
│   ├── Train_Model.ipynb           # Classifier benchmarking (RF, SVM-RBF, MLP) & held-out evaluation
│   ├── virtual_mouse_brain.pkl     # Deployed SVM pipeline (StandardScaler + SVC, 1,139 SVs)
│   ├── confusion_matrix_Support_Vector_Machine_(SVM).png
│   ├── confusion_matrix_Random_Forest.png
│   └── confusion_matrix_Neural_Network_(MLP).png
│
├── Test/
│   ├── benchmark_latency.py        # Per-stage pipeline latency profiler (500 frames)
│   ├── mouse_audit.txt             # Primary 200-event operational telemetry trace
│   ├── audit_baseline.txt          # Ablation log: All controls disabled
│   ├── audit_wo_freeze.txt         # Ablation log: Coordinate freeze disabled (55 lines)
│   ├── audit_wo_debounce.txt       # Ablation log: Wall-clock debounce disabled (116 lines)
│   ├── audit_wo_latch.txt          # Ablation log: Context-menu latch disabled (109 lines)
│   ├── audit_wo_drag.txt           # Ablation log: Two-frame drag hysteresis disabled
│   ├── audit_proposed.txt          # Ablation log: Full proposed pipeline enabled
│   ├── p1.txt                      # 20-event telemetry snapshot: Unseen Subject 1
│   ├── p2.txt                      # 20-event telemetry snapshot: Unseen Subject 2
│   └── p3.txt                      # 20-event telemetry snapshot: Unseen Subject 3
│
├── Utils/
│   ├── HandTrackingModule.py       # MediaPipe HandLandmarker wrapper + temporal wrist anchor tracker
│   └── hand_landmarker.task        # MediaPipe model asset
│
├── .gitignore
├── mouse_audit.txt                 # Default runtime export trace
├── README.md
└── requirements.txt
~~~

---

## Installation

### Prerequisites

| Requirement | Version |
|:------------|:--------|
| Python | 3.10+ (Tested on Python 3.13.5) |
| OS | Windows 10/11 (primary testbed: Windows 11 Build 26200), macOS / Linux |
| Hardware | Monocular USB Webcam, 4+ CPU cores recommended |

### Setup

~~~bash
# 1. Clone the repository
git clone https://github.com/HAITHM-ALAKHALI/virtual_mouse.git
cd virtual_mouse

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
~~~

### Dependencies

| Package | Purpose |
|:--------|:--------|
| `mediapipe` | Hand landmark detection (Tasks API, `RunningMode.VIDEO`) |
| `opencv-python` | Camera capture and image processing |
| `scikit-learn` | SVM classifier pipeline (`StandardScaler` + `SVC`) |
| `pynput` | OS-level mouse event injection (`Controller`, `Button`) |
| `customtkinter` | Modern dark-themed GUI framework |
| `Pillow` | BGR to RGB image conversion for GUI preview |
| `numpy` | Vectorized array operations and linear interpolation |
| `joblib` | Model serialization and loading |

---

## Quick Start & Reproducing Experiments

### 1. Run the Live Application

~~~bash
cd Agent_Execution
python App_GUI.py
~~~

Click **▶ Start Virtual Mouse** in the GUI and position your hand in front of the webcam.

### 2. Reproduce Ablation Study Configurations

Inside `Agent_Execution/Main.py`, toggle the boolean ablation flags inside `VirtualMouseController.__init__()` to isolate individual dispatcher subsystems:

~~~python
self.ENABLE_COORD_FREEZE = True      # Lock (plocX, plocY) during click actuation
self.ENABLE_TIME_DEBOUNCE = True     # Wall-clock 0.25s left-click refractory gate
self.ENABLE_MENU_LATCH = True        # State-latched context-menu retention (L_R)
self.ENABLE_DRAG_HYSTERESIS = True   # 2-frame persistence filter before drag drop
self.ENABLE_SCROLL_DEADZONE = True   # Vertical center deadband for scroll gestures
self.ENABLE_SMOOTHING = True         # Exponential cursor smoothing divisor
~~~

Click **📥 Export Log** in the GUI after any session to export the complete millisecond-stamped trace.

### 3. Collect Custom Training Data

~~~bash
cd Data_Engineering/Dataset
python Collect_Data.py
~~~

Hold a gesture and press the corresponding key (`m`: Move, `l`: Left_Click, `r`: Right_Click, `s`: Scroll, `d`: Drag, `n`: Neutral, `q`: Quit) to append labeled 42D feature vectors to `virtual_mouse_data.csv`.

### 4. Run the Latency Benchmark

~~~bash
cd Test
python benchmark_latency.py
~~~

Profiles 500 consecutive operational cycles across MediaPipe inference, 42D normalization, SVM classification, and action dispatch overhead.

---

## Dataset & Pre-trained Weights

| Asset | Location | Description |
|:------|:---------|:------------|
| Development dataset | [`Data_Engineering/Dataset/virtual_mouse_data.csv`](Data_Engineering/Dataset/virtual_mouse_data.csv) | 10,008 instances × 42 features + label (5 subjects) |
| External held-out cohort | [`Data_Engineering/Dataset/p1.csv`](Data_Engineering/Dataset/p1.csv), [`p2.csv`](Data_Engineering/Dataset/p2.csv), [`p3.csv`](Data_Engineering/Dataset/p3.csv) | 766 unseen instances across 3 external subjects |
| Kaggle dataset | [kaggle.com/datasets/haithmalakhali/virtual-mouse-landmarks](https://www.kaggle.com/datasets/haithmalakhali/virtual-mouse-landmarks) | Public Release v1.0.0 |
| Trained model | [`Model_Training/virtual_mouse_brain.pkl`](Model_Training/virtual_mouse_brain.pkl) | `Pipeline([StandardScaler, SVC(rbf, C=10.0)])` — 1,139 SVs |

---

## Gesture Reference & Runtime Configuration

| Gesture | Hand Pose | Rule-Based OS Action |
|:--------|:----------|:---------------------|
| ☝️ **Move** | Index finger extended | Continuous cursor tracking with smoothing divisor `alpha` |
| ✌️ **Left Click** | Index + Middle fingers | Coordinate-frozen, time-gated left click (0.25 s debounce) |
| 🤟 **Right Click** | Index + Middle + Ring | Coordinate-frozen, state-latched right click (single dispatch) |
| 🖐️ **Scroll** | Open palm | Bidirectional vertical scroll outside center deadzone |
| ✊ **Drag** | Closed fist | Press-hold `Button.left`; released after 2-frame hysteresis |
| 🤚 **Neutral** | Relaxed / Transition pose | Suppresses OS event injection and resets non-latched states |

---

## Citation

If you use this architecture, code, or dataset in your research, please cite:

~~~bibtex
@inproceedings{alakhali2026hybrid,
  author    = {Al-Akhali, Haithm Abdulwali},
  title     = {A Hybrid Architecture for Vision-Based Virtual Mouse Systems: Normalized Gesture Classification and Deterministic OS Event Dispatching},
  booktitle = {IEEE Conference Proceedings / Preprint},
  year      = {2026},
  url       = {https://github.com/HAITHM-ALAKHALI/virtual_mouse}
}
~~~

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Author

**Haithm Abdulwali Al-Akhali**  
Department of Computer Science, Faculty of Computer Science and Information Technology  
Sana'a University, Sana'a, Yemen  

[🌐 Portfolio](https://haithm-al-akhali.vercel.app/) · [💻 GitHub](https://github.com/HAITHM-ALAKHALI) · [📊 Kaggle](https://www.kaggle.com/datasets/haithmalakhali/virtual-mouse-landmarks) · [✉️ Email](mailto:haithmabdulwali.alakhali@gmail.com)