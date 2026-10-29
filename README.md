<div align="center">

# A Hybrid Architecture for Vision-Based Virtual Mouse Systems

_Normalized Gesture Classification and Rule-Based OS Event Dispatching_

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-00A98F?style=flat-square&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle_v1.0.0-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/haithmalakhali/virtual-mouse-landmarks)
[![Preprint](https://img.shields.io/badge/arXiv-Preprint-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/)
[![IEEE Paper](https://img.shields.io/badge/Paper-IEEE_Format-00629B?style=flat-square&logo=ieee&logoColor=white)](https://github.com/HAITHM-ALAKHALI/virtual_mouse)

</div>

---

## Abstract

This repository contains the official reference implementation, benchmark scripts, and empirical telemetry logs for the research paper:

> **A Hybrid Architecture for Vision-Based Virtual Mouse Systems: Normalized Gesture Classification and Rule-Based OS Event Dispatching**  
> _Haithm Abdulwali Al-Akhali — Department of Computer Science, Faculty of Computer Science and Information Technology, Sana'a University_

Vision-based human-computer interaction (HCI) interfaces frequently suffer from operational instability and actuation ambiguity, including high-frequency cursor jitter, unintentional multi-click triggers, and unintended context-menu dismissals caused by hardware-dependent frame rate variations and involuntary physiological hand tremors. This project presents an empirically evaluated hybrid architecture for real-time virtual mouse control that bridges probabilistic computer vision inference with rule-based operating system (OS) event-dispatching constraints.

The proposed pipeline decouples spatial landmark kinematics from intentional state recognition: 21 three-dimensional hand landmarks are extracted via MediaPipe Tasks in continuous video mode, stabilized using a temporal wrist-anchoring state machine to reduce unintended target-hand switching. Retained 2D image-plane components undergo translation normalization, approximate uniform-scale normalization, and handedness canonicalization via horizontal reflection. Evaluated on a curated dataset of **10,008 empirical instances** collected across five human subjects and six operational classes (`Move`, `Left_Click`, `Right_Click`, `Drag`, `Scroll`, and `Neutral`), an **RBF-kernel Support Vector Machine (SVM)** pipeline achieves **98.10% sample-level test accuracy** under a stratified random split with an isolated inference latency of **0.717 ms** (averaging **0.72 ms** under steady-state daemon execution), paired with rapid model training time (**0.68 s**) enabling fast local model retraining. Crucially, the frozen classifier pipeline (including fitted feature scaling) was independently evaluated in a zero-shot evaluation on three unseen human subjects (**766 sequential test frames**), achieving an overall cross-subject accuracy of **88.12%** and macro F1-score of **0.88** without subject-specific retraining. Complementary live interactive pilot sessions across 10 human participants demonstrated operational usability across the primary interaction modes under the tested desktop conditions. Downstream, an FPS-decoupled, wall-clock rate-limiting mechanism ($0.25\text{ s}$ refractory gate) combined with coordinate freezing, an event-latched state machine, and two-frame drag hysteresis regulates actuation frequency, retains context menus, and maintains stable logged actuation coordinates across exploratory live ablation trials under the tested conditions. Furthermore, a thread-safe circular telemetry buffer isolates vision processing from GUI rendering, delivering continuous millisecond-resolution audit tracing for both dispatched actions and suppressed event candidates, with an average post-detection pipeline overhead of approximately **1.22 ms**.

---

## Key Architectural Features

- **Temporal Target-Hand Anchoring (`Utils/HandTrackingModule.py`)**
  - Wrist-based hand identity tracking with Euclidean distance threshold $D_{\max} = 0.25$ in normalized image space.
  - Tracking-loss hysteresis counter retaining anchor parameters for up to $M_{\text{lost}} = 20$ consecutive frames to prevent disruptive target re-acquisitions.

- **Bifurcated 42D Anatomical Normalization Pipeline**
  - **Continuous Kinematic Stream:** Maps raw index fingertip $\mathbf{p}_8 = (x_8, y_8)$ to display resolution $(w_{\text{scr}}, h_{\text{scr}})$ with an exponential smoothing divisor $\alpha \in [1, 15]$.
  - **Normalized Classification Stream:**
    1. _Translation Normalization:_ Wrist landmark ($\mathbf{p}_0$) defined as the coordinate origin ($\mathbf{q}_i = \mathbf{p}_i - \mathbf{p}_0$).
    2. _Anatomical Scale Normalization:_ Scaled by characteristic palm length $s = \Vert{}\mathbf{p}_9 - \mathbf{p}_0\Vert{}_2$ with an explicit division-by-zero safeguard.
    3. _Handedness Canonicalization:_ Horizontal reflection ($\hat{x}_i \leftarrow -\hat{x}_i$ if $\text{Handedness} = \text{Left}$) canonicalizing left-hand geometry into the right-hand convention.

- **Rule-Based Action Dispatcher (`Agent_Execution/Main.py`)**
  - **FPS-Decoupled Wall-Clock Rate Limiting:** Enforces $\Delta t = t_{\text{now}} - t_{\text{last}} \ge T_{\text{interval}}$ using `time.time()`, where $T_{\text{interval}} = \max(0.05, \text{clickDelay} \times 0.025)\text{ s}$ (defaulting to $0.25\text{ s}$ at slider value 10).
  - **Actuation Coordinate Freezing:** Explicitly locks OS pointer coordinates at the pre-actuation anchor $(p_{\text{locX}}, p_{\text{locY}})$ during `Left_Click` and `Right_Click` actuation to mitigate physical finger-flexion slip.
  - **Latched Context-Menu State Machine ($L_R$):** Dispatches a single right-click event and latches state, suppressing redundant frame triggers until an explicit non-right-click gesture transition or tracking loss occurs.
  - **Two-Frame Drag Hysteresis ($\Delta f \ge 2$):** Requires two consecutive non-drag frames before releasing `Button.left`, absorbing transient single-frame classification noise.
  - **Scroll Deadzone Engine:** Evaluates vertical displacement relative to a configurable vertical deadzone ($\vert{}y_8 - Y_{\text{center}}\vert{} > \text{deadzone}$) with dynamic directional velocity scaling.

- **Dual-Thread Asynchronous Telemetry & Full Session Auditing**
  - Vision acquisition, inference, and OS dispatching execute in an isolated daemon thread decoupled from the `CustomTkinter` GUI main loop.
  - Protected by synchronization primitives (`_frame_lock` and `_log_lock`) with an in-memory preview queue (`deque(maxlen=20)`) and full-session export buffers logging both dispatched actions and `SUPPRESSED` candidates at millisecond resolution.

---

## Experimental Benchmarks & Results

### 1. Classifier Pipeline Comparison (2,002 Test Instances — Stratified 80/20 Split)

| Classifier Architecture            | Accuracy (%) | Macro Precision | Macro Recall | Macro F1-Score | Isolated Latency (ms) | Training Time (s) |
| :--------------------------------- | :----------: | :-------------: | :----------: | :------------: | :-------------------: | :---------------: |
| Random Forest (100 Estimators)     |    97.35%    |      0.97       |     0.97     |      0.97      |         9.117         |       4.43        |
| **SVM (RBF Kernel, $C=10.0$)**     |  **98.10%**  |    **0.98**     |   **0.98**   |    **0.98**    |       **0.717**       |     **0.68**      |
| Neural Network (MLP, 64$\times$32) |    98.20%    |      0.98       |     0.98     |      0.98      |         0.591         |       14.66       |

### 2. Zero-Shot Cross-Subject Generalization (External Held-Out Cohort)

Evaluated directly using the frozen `virtual_mouse_brain.pkl` pipeline on sequential frames from three unseen human participants without any fine-tuning or calibration:

| Evaluation Cohort            |    Subjects    | Dataset File             | Sample Count | Accuracy (%) | Macro F1-Score |
| :--------------------------- | :------------: | :----------------------- | :----------: | :----------: | :------------: |
| Internal Test Split          |   5 (Shared)   | `virtual_mouse_data.csv` |    2,002     |    98.10%    |      0.98      |
| Subject 1 (External)         |   1 (Unseen)   | `p1.csv`                 |     292      |    81.51%    |       —        |
| Subject 2 (External)         |   1 (Unseen)   | `p2.csv`                 |     212      |   100.00%    |       —        |
| Subject 3 (External)         |   1 (Unseen)   | `p3.csv`                 |     262      |    85.88%    |       —        |
| **Combined Held-Out Cohort** | **3 (Unseen)** | `p1.csv` – `p3.csv`      |   **766**    |  **88.12%**  |    **0.88**    |

### 3. Exploratory Live Ablation Study of Action Dispatcher Mechanisms

Conducted across 6 independent live interactive camera sessions isolating each software control in `Main.py` and auditing raw internal telemetry traces in `Test/`:

| Configuration Pipeline                 | Audit Log File (`Test/`)                              | Logged Drift Span (px) | Cascade Log Events | Redundant Menu Events | Premature Drops (Drops / Starts) |
| :------------------------------------- | :---------------------------------------------------- | :--------------------: | :----------------: | :-------------------: | :------------------------------: |
| _Naive Baseline (All Off)_             | [`audit_baseline.txt`](Test/audit_baseline.txt)       |         99.92          |   59 (57 excess)   |    48 (47 excess)     |         1 / 1 (100.0\%)          |
| w/o Actuation Coord Freeze             | [`audit_wo_freeze.txt`](Test/audit_wo_freeze.txt)     |         54.41          |         0          |           0           |          0 / 0 (0.0\%)           |
| w/o Rate Limiting ($\Delta t$)         | [`audit_wo_debounce.txt`](Test/audit_wo_debounce.txt) |        **0.00**        |   72 (71 excess)   |           0           |          0 / 0 (0.0\%)           |
| w/o Latched Menu FSM ($L_R$)           | [`audit_wo_latch.txt`](Test/audit_wo_latch.txt)       |        **0.00**        |         0          |     61 (60 seq.)      |          0 / 0 (0.0\%)           |
| w/o Drag Hysteresis ($\Delta f \ge 2$) | [`audit_wo_drag.txt`](Test/audit_wo_drag.txt)         |        **0.00**        |         0          |           0           |         7 / 12 (58.3\%)          |
| **Full Proposed Pipeline**             | [`audit_proposed.txt`](Test/audit_proposed.txt)       |        **0.00**        |       **0**        |         **0**         |        **0 / 1 (0.0\%)**         |

> **Methodological Note:** Because each ablation condition was evaluated in a separate live webcam session, the recorded event counts and coordinate displacements represent descriptive telemetry observations under the tested conditions rather than trial-matched causal estimates from identical video replays. In `audit_proposed.txt`, the full configuration logged 15 accepted `LEFT_CLICK` actuations (52 cooldown candidates suppressed), 2 accepted `RIGHT_CLICK` dispatches (76 held-state candidates suppressed), 1 `DRAG_START`, 1 completed `DRAG_DROP` (0 premature drops), and 8 `SCROLL_UP` events (13 deadzone candidates suppressed).

### 4. End-to-End Runtime Latency Profile (500 Operational Cycles)

| Pipeline Stage                                 | Mean Latency $\pm$ Std (ms) | Notes                                       |
| :--------------------------------------------- | :-------------------------: | :------------------------------------------ |
| 1. Video Acquisition & MediaPipe Inference     |    33.53 $\pm$ 38.54 ms     | 95th percentile: 62.93 ms                   |
| 2. 42D Anatomical Normalization                |     0.06 $\pm$ 0.01 ms      | Translation, scale, handedness reflection   |
| 3. SVM-RBF Classification                      |     0.72 $\pm$ 0.14 ms      | 0.717 ms isolated micro-benchmark           |
| 4. Action Dispatcher & OS Event Injection      |     0.44 $\pm$ 0.08 ms      | Includes mutex telemetry logging + `pynput` |
| **Total Post-Detection Overhead (Stages 2–4)** |        **~1.22 ms**         | **$< 4\%$ of nominal 30 FPS frame budget**  |

---

## Repository Structure

```text
├── Agent_Execution/
│   ├── App_GUI.py                  # Dark-themed control panel & full-session log exporter (CustomTkinter)
│   └── Main.py                     # VirtualMouseController — daemon thread pipeline & ablation flags
│
├── Data_Engineering/
│   ├── Collect_Data.py             # Live landmark data collection with key-labelling
│   ├── virtual_mouse_data.csv      # 10,008 labelled development instances (5 subjects)
│   ├── p1.csv                      # Unseen held-out Subject 1 dataset (292 instances)
│   ├── p2.csv                      # Unseen held-out Subject 2 dataset (212 instances)
│   ├── p3.csv                      # Unseen held-out Subject 3 dataset (262 instances)
│
├── Model_Training/
│   ├── Train_Model.ipynb           # Classifier benchmarking (RF, SVM-RBF, MLP) & held-out evaluation
│   ├── virtual_mouse_brain.pkl     # Deployed SVM pipeline (StandardScaler + SVC, 1,139 SVs)
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_rf.png
│   └── confusion_matrix_mlp.png
│
├── Test/
│   ├── benchmark_latency.py        # Per-stage pipeline latency profiler (500 frames)
│   ├── mouse_audit.txt             # Primary 456-event operational telemetry trace
│   ├── audit_baseline.txt          # Ablation log: All controls disabled (118 entries)
│   ├── audit_wo_freeze.txt         # Ablation log: Coordinate freeze disabled (57 entries)
│   ├── audit_wo_debounce.txt       # Ablation log: Wall-clock rate limiting disabled (116 entries)
│   ├── audit_wo_latch.txt          # Ablation log: Context-menu latch disabled (118 entries)
│   ├── audit_wo_drag.txt           # Ablation log: Two-frame drag hysteresis disabled (60 entries)
│   ├── audit_proposed.txt          # Ablation log: Full proposed pipeline enabled (168 entries)
│   ├── p1.txt                      # 21-event telemetry snapshot: Unseen Subject 1
│   ├── p2.txt                      # 22-event telemetry snapshot: Unseen Subject 2
│   └── p3.txt                      # 21-event telemetry snapshot: Unseen Subject 3
│
├── Utils/
│   ├── HandTrackingModule.py       # MediaPipe HandLandmarker wrapper + temporal wrist anchor tracker
│   └── hand_landmarker.task        # MediaPipe model asset
│
├── .gitignore
├── CITATION.cff                    # Citation metadata for academic indexing
├── LICENSE                         # MIT License file
├── mouse_audit.txt                 # Default runtime export trace
├── README.md
└── requirements.txt
```

---

## Installation

### Prerequisites

| Requirement | Version                                                                |
| :---------- | :--------------------------------------------------------------------- |
| Python      | 3.10+ (Tested on Python 3.13.5)                                        |
| OS          | Windows 10/11 (primary testbed: Windows 11 Build 26200), macOS / Linux |
| Hardware    | Monocular USB Webcam, 4+ CPU cores recommended                         |

### Setup

```bash
# 1. Clone the repository
git clone [https://github.com/HAITHM-ALAKHALI/virtual_mouse.git](https://github.com/HAITHM-ALAKHALI/virtual_mouse.git)
cd virtual_mouse

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package         | Purpose                                                  |
| :-------------- | :------------------------------------------------------- |
| `mediapipe`     | Hand landmark detection (Tasks API, `RunningMode.VIDEO`) |
| `opencv-python` | Camera capture and image processing                      |
| `scikit-learn`  | SVM classifier pipeline (`StandardScaler` + `SVC`)       |
| `pynput`        | OS-level mouse event injection (`Controller`, `Button`)  |
| `customtkinter` | Modern dark-themed GUI framework                         |
| `Pillow`        | BGR to RGB image conversion for GUI preview              |
| `numpy`         | Vectorized array operations and linear interpolation     |
| `joblib`        | Model serialization and loading                          |

---

## Quick Start & Reproducing Experiments

### 1. Run the Live Application

```bash
cd Agent_Execution
python App_GUI.py
```

1. Click **▶ Start Virtual Mouse** in the GUI.
2. Adjust runtime calibration sliders (**Smoothening**, **Scroll Deadzone**, **Scroll Speed**, **Click Delay**) to customize interaction dynamics.
3. Position your hand within the camera frame.

### 2. Reproduce Ablation Study Configurations

Inside `Agent_Execution/Main.py`, toggle the boolean ablation flags inside `VirtualMouseController.__init__()` to isolate individual dispatcher subsystems:

```python
self.ENABLE_COORD_FREEZE = True      # Lock (plocX, plocY) during click actuation
self.ENABLE_TIME_DEBOUNCE = True     # Wall-clock left-click refractory gate (Delta_t)
self.ENABLE_MENU_LATCH = True        # State-latched context-menu retention (L_R)
self.ENABLE_DRAG_HYSTERESIS = True   # 2-frame persistence filter before drag drop
self.ENABLE_SCROLL_DEADZONE = True   # Vertical center deadband for scroll gestures
self.ENABLE_SMOOTHING = True         # Exponential cursor smoothing divisor
```

Click **📥 Export Log** in the GUI after any session to export the complete millisecond-stamped trace.

### 3. Collect Custom Training Data

```bash
cd Data_Engineering
python Collect_Data.py
```

Hold a gesture and press the corresponding key (`m`: Move, `l`: Left_Click, `r`: Right_Click, `s`: Scroll, `d`: Drag, `n`: Neutral, `q`: Quit) to append labeled 42D feature vectors to `virtual_mouse_data.csv`.

### 4. Run the Latency Benchmark

```bash
cd Test
python benchmark_latency.py
```

Profiles 500 consecutive operational cycles across MediaPipe inference, 42D normalization, SVM classification, and action dispatch overhead.

---

## Dataset & Pre-trained Weights

| Asset                    | Location                                                                                                                             | Description                                                |
| :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------- |
| Development dataset      | [`Data_Engineering/virtual_mouse_data.csv`](Data_Engineering/virtual_mouse_data.csv)                                                 | 10,008 instances $\times$ 42 features + label (5 subjects) |
| External held-out cohort | [`Data_Engineering/p1.csv`](Data_Engineering/p1.csv), [`p2.csv`](Data_Engineering/p2.csv), [`p3.csv`](Data_Engineering/p3.csv)       | 766 unseen instances across 3 external subjects            |
| Kaggle dataset           | [kaggle.com/datasets/haithmalakhali/virtual-mouse-landmarks](https://www.kaggle.com/datasets/haithmalakhali/virtual-mouse-landmarks) | Public Release v1.0.0                                      |
| Trained model            | [`Model_Training/virtual_mouse_brain.pkl`](Model_Training/virtual_mouse_brain.pkl)                                                   | `Pipeline([StandardScaler, SVC(rbf, C=10.0)])` — 1,139 SVs |

---

## Gesture Reference & Runtime Configuration

| Gesture            | Hand Pose                 | Rule-Based OS Action                                                               |
| :----------------- | :------------------------ | :--------------------------------------------------------------------------------- |
| ☝️ **Move**        | Index finger extended     | Continuous cursor tracking with smoothing divisor $\alpha$                         |
| ✌️ **Left Click**  | Index + Middle fingers    | Coordinate-frozen, rate-limited left click ($T_{\text{interval}} = 0.25\text{ s}$) |
| 🤟 **Right Click** | Index + Middle + Ring     | Coordinate-frozen, state-latched right click (single dispatch)                     |
| 🖐️ **Scroll**      | Open palm                 | Bidirectional vertical scroll outside center deadzone                              |
| ✊ **Drag**        | Closed fist               | Press-hold `Button.left`; released after 2-frame hysteresis                        |
| 🤚 **Neutral**     | Relaxed / Transition pose | Suppresses OS event injection and resets non-latched states                        |

---

## Citation

If you use this architecture, code, or dataset in your research, please cite:

```bibtex
@inproceedings{alakhali2026hybrid,
  author    = {Al-Akhali, Haithm Abdulwali},
  title     = {A Hybrid Architecture for Vision-Based Virtual Mouse Systems: Normalized Gesture Classification and Rule-Based OS Event Dispatching},
  booktitle = {IEEE Conference Proceedings / arXiv Preprint},
  year      = {2026},
  url       = {[https://github.com/HAITHM-ALAKHALI/virtual_mouse](https://github.com/HAITHM-ALAKHALI/virtual_mouse)}
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

[🌐 Portfolio](https://haithm-al-akhali.vercel.app/) · [💻 GitHub](https://github.com/HAITHM-ALAKHALI) · [📊 Kaggle](https://www.kaggle.com/datasets/haithmalakhali/virtual-mouse-landmarks) · [✉️ Email](mailto:haithmabdulwali.alakhali@gmail.com)
