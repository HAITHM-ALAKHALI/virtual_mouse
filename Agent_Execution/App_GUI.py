"""
================================================================================
App_GUI.py — AI Virtual Mouse Control Panel & Telemetry Dashboard
Virtual Mouse Control System
Paper Reference: Section III-C ("Rule-Based Action Dispatching Engine")
                 Section III-D ("Thread Concurrency and Telemetry Logging")
                 Section IV-F ("Operational Consistency and Reliability Audits")
================================================================================

Description:
    Asynchronous graphical dashboard constructed with CustomTkinter.
    Decouples UI event rendering from real-time computer vision acquisition
    and OS action dispatching executing on an isolated background daemon thread.

Core Architectural Features:
    1. Asynchronous Multithreading (Section III-D):
       Host GUI loop executes independently from `VirtualMouseController`,
       guaranteeing responsive rendering (~33 FPS refresh) without thread stalls.
    2. Operational Calibration Sliders (Section III-B & III-C):
       - Smoothening: Dynamically tunes exponential divisor alpha in [1, 15] (Eq. 4).
       - Click Delay: Configures wall-clock gate T_interval = max(0.05, val * 0.025) s.
       - Scroll Deadzone: Adjusts vertical deadband threshold |Delta_Y| > deadzone.
       - Scroll Speed: Adjusts dynamic scroll displacement step multipliers.
    3. Live Event Auditing & Full Session Exporter (Section III-D & IV-F):
       Displays thread-safe circular preview buffers (deque maxlen=20) and exposes
       an on-demand session exporter generating millisecond-stamped traces (mouse_audit.txt).
================================================================================
"""

import customtkinter as ctk
import threading
import webbrowser
import logging
import cv2
from PIL import Image, ImageTk
from Main import VirtualMouseController
from tkinter import filedialog, messagebox
from datetime import datetime

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global Appearance & Theme Configuration
# -----------------------------------------------------------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Consistent Dark-Mode Visual Palette
BG_DARK = "#111111"
CARD_BG = "#191919"
CARD_BORDER = "#2A2A2A"

ACCENT = "#4587A5"
ACCENT_HOVER = "#366B84"

SUCCESS = "#4ADE80"
SUCCESS_HOVER = "#22C55E"

DANGER = "#F87171"
DANGER_HOVER = "#EF4444"

TEXT_PRIMARY = "#F5F5F5"
TEXT_SECONDARY = "#A3A3A3"
TEXT_MUTED = "#737373"

DIVIDER = "#262626"

# Gesture state visual color coding (matches camera OpenCV HUD overlays)
GESTURE_COLORS = {
    "Move": "#ff00ff",
    "Left_Click": "#00ff00",
    "Right_Click": "#ff4444",
    "Scroll": "#ffff00",
    "Drag": "#00ffff",
}


class VirtualMouseApp(ctk.CTk):
    """
    Main Application Window encapsulating the asynchronous GUI dashboard,
    real-time camera preview, operational tuning sliders, and telemetry audit exporter.
    """

    PREVIEW_W = 640
    PREVIEW_H = 500
    UPDATE_INTERVAL_MS = (
        30  # ~33 FPS UI refresh rate to decouple rendering from vision pipeline
    )

    def __init__(self) -> None:
        super().__init__()

        # ── Window Setup ──
        self.title("AI Virtual Mouse")
        self.geometry("750x780")
        self.minsize(700, 780)
        self.resizable(False, False)
        self.configure(fg_color=BG_DARK)

        # ── Main Scrollable Viewport Container ──
        self.scroll_container = ctk.CTkScrollableFrame(
            self,
            fg_color=BG_DARK,
            scrollbar_button_color=CARD_BORDER,
            scrollbar_button_hover_color=TEXT_MUTED,
        )
        self.scroll_container.pack(fill="both", expand=True)

        # ---------------------------------------------------------------------
        # Daemon Controller & Worker Thread State (Section III-D)
        # ---------------------------------------------------------------------
        self.controller = VirtualMouseController()
        self.mouse_thread = None
        self._is_active = False
        self._preview_job = None

        # ── UI Construction Pipeline ──
        self._build_header()
        self._build_camera_preview()
        self._build_status_bar()
        self._build_settings()
        self._build_event_log()
        self._build_gesture_guide()
        self._build_footer()

        # Intercept window manager close event for safe thread teardown
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # UI Component Construction
    # ==================================================================

    def _build_header(self) -> None:
        """Renders top application title and release version banner."""
        header = ctk.CTkFrame(self.scroll_container, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(12, 0))

        ctk.CTkLabel(
            header,
            text="🖱  AI Virtual Mouse",
            font=("Segoe UI", 22, "bold"),
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text="v2.0",
            font=("Segoe UI", 12),
            text_color=TEXT_MUTED,
        ).pack(side="left", padx=(8, 0), pady=(6, 0))

    # ── Camera Preview ──────────────────────────────────────────────

    def _build_camera_preview(self) -> None:
        """Constructs camera viewport card container and placeholder."""
        self.preview_card = ctk.CTkFrame(
            self.scroll_container,
            fg_color=CARD_BG,
            corner_radius=14,
            border_width=1,
            border_color=CARD_BORDER,
            width=720,  # Fixed container width
            height=520,
        )
        self.preview_card.pack(fill="none", padx=24, pady=(8, 0))
        self.preview_card.pack_propagate(False)  # Enforce fixed layout bounds

        # Centered preview display canvas for rendered OpenCV frames
        self.preview_label = ctk.CTkLabel(
            self.preview_card,
            text="",
            width=700,
            height=500,
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        self._set_preview_placeholder()

    def _set_preview_placeholder(self) -> None:
        """Displays standby notification text when camera pipeline is idle."""
        self.preview_label.configure(
            text="📷  Camera preview will appear here\n(Note: The camera may take a while to open)",
            font=("Segoe UI", 14),
            text_color=TEXT_MUTED,
            image=None,
        )

    # ── Status Bar ───────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        """Constructs operational status indicators, FPS counter, and daemon toggle button."""
        bar = ctk.CTkFrame(self.scroll_container, fg_color="transparent")
        bar.pack(fill="x", padx=24, pady=(8, 0))

        # Status indicator dot and text
        status_left = ctk.CTkFrame(bar, fg_color="transparent")
        status_left.pack(side="left")

        self.status_dot = ctk.CTkLabel(
            status_left,
            text="●",
            font=("Segoe UI", 14),
            text_color=TEXT_MUTED,
            width=20,
        )
        self.status_dot.pack(side="left")

        self.status_label = ctk.CTkLabel(
            status_left,
            text="System Ready",
            font=("Segoe UI", 13),
            text_color=TEXT_SECONDARY,
        )
        self.status_label.pack(side="left", padx=(4, 0))

        # Runtime FPS and active classified gesture display
        info_right = ctk.CTkFrame(bar, fg_color="transparent")
        info_right.pack(side="right")

        self.fps_label = ctk.CTkLabel(
            info_right,
            text="FPS: —",
            font=("Segoe UI Semibold", 12),
            text_color=TEXT_MUTED,
        )
        self.fps_label.pack(side="left", padx=(0, 16))

        self.action_label = ctk.CTkLabel(
            info_right,
            text="Action: —",
            font=("Segoe UI Semibold", 12),
            text_color=TEXT_MUTED,
        )
        self.action_label.pack(side="left")

        # Daemon loop toggle button (Start / Stop)
        self.btn_toggle = ctk.CTkButton(
            self.scroll_container,
            text="▶   Start Virtual Mouse",
            font=("Segoe UI Semibold", 14),
            fg_color=SUCCESS,
            hover_color=SUCCESS_HOVER,
            text_color="#ffffff",
            height=42,
            corner_radius=10,
            command=self._toggle,
        )
        self.btn_toggle.pack(pady=(8, 0), padx=24, fill="x")

    # ── Settings ─────────────────────────────────────────────────────

    def _build_settings(self) -> None:
        """
        Constructs real-time calibration slider rows exposed to the user
        (Section III-B & III-C in paper).
        """
        ctk.CTkLabel(
            self.scroll_container,
            text="⚙  Settings",
            font=("Segoe UI Semibold", 14),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).pack(fill="x", padx=28, pady=(10, 0))

        card = ctk.CTkFrame(
            self.scroll_container,
            fg_color=CARD_BG,
            corner_radius=12,
            border_width=1,
            border_color=CARD_BORDER,
        )
        card.pack(fill="x", padx=24, pady=(4, 0))

        # 1. Smoothening divisor alpha in [1, 15] (Paper Equation 4)
        self.lbl_smooth = self._make_slider_row(
            card, "Smoothening", 1, 15, 7, self._on_smooth, row=0
        )
        # 2. Vertical scroll center deadzone threshold (Section III-C)
        self.lbl_deadzone = self._make_slider_row(
            card, "Scroll Deadzone", 20, 150, 60, self._on_deadzone, row=1
        )
        # 3. Dynamic scroll speed scaling multiplier
        self.lbl_scroll_speed = self._make_slider_row(
            card, "Scroll Speed", 1, 10, 3, self._on_scroll_speed, row=2
        )
        # 4. Click delay refractory gate multiplier: T_interval = max(0.05, clickDelay * 0.025) s
        self.lbl_clickDelay = self._make_slider_row(
            card,
            "Click Delay",
            0,
            20,
            self.controller.clickDelay,
            self._on_clickDelay,
            row=3,
        )

    def _make_slider_row(self, parent, label, from_, to, default, command, row):
        """Constructs an accessible slider row with dynamic numeric readouts."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=16, pady=(6 if row == 0 else 2, 6 if row == 3 else 2))

        name_lbl = ctk.CTkLabel(
            frame,
            text=label,
            font=("Segoe UI", 12),
            text_color=TEXT_SECONDARY,
            width=130,
            anchor="w",
        )
        name_lbl.pack(side="left")

        val_lbl = ctk.CTkLabel(
            frame,
            text=str(int(default)),
            font=("Segoe UI Semibold", 12),
            text_color=TEXT_PRIMARY,
            width=36,
            anchor="e",
        )

        slider = ctk.CTkSlider(
            frame,
            from_=from_,
            to=to,
            number_of_steps=int(to - from_),
            width=320,
            height=16,
            fg_color=DIVIDER,
            progress_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
        )
        slider.set(default)
        slider.pack(side="left", padx=(8, 8))
        val_lbl.pack(side="right")

        # Synchronous callback updating value label and mutating controller property
        def _update_value(*args):
            current_val = int(round(slider.get()))
            val_lbl.configure(text=str(current_val))
            if command:
                command(current_val)

        # 1. Bind slider position change callback
        slider.configure(command=lambda _: _update_value())

        # 2. Force immediate mouse-motion capture across track events
        try:
            internal_canvas = slider._canvas
            internal_canvas.bind(
                "<B1-Motion>", lambda e: self.after_idle(_update_value), add="+"
            )
            internal_canvas.bind(
                "<Button-1>", lambda e: self.after_idle(_update_value), add="+"
            )
            internal_canvas.bind(
                "<ButtonRelease-1>", lambda e: self.after_idle(_update_value), add="+"
            )
        except AttributeError:
            slider.bind(
                "<B1-Motion>", lambda e: self.after_idle(_update_value), add="+"
            )
            slider.bind("<Button-1>", lambda e: self.after_idle(_update_value), add="+")

        return val_lbl

    # ── Live Telemetry Log & Full Session Exporter ────────────────────

    def _build_event_log(self) -> None:
        """
        Constructs the live event audit panel displaying the circular preview queue
        and full-session exporter button (Paper Section III-D & IV-F).
        """
        header_frame = ctk.CTkFrame(self.scroll_container, fg_color="transparent")
        header_frame.pack(fill="x", padx=28, pady=(10, 0))

        ctk.CTkLabel(
            header_frame,
            text="📋  Event Audit Log (Action Dispatcher Trace)",
            font=("Segoe UI Semibold", 14),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).pack(side="left")

        # Session export trigger button
        ctk.CTkButton(
            header_frame,
            text="📥 Export Log",
            font=("Segoe UI", 11, "bold"),
            fg_color=CARD_BG,
            hover_color=CARD_BORDER,
            text_color=ACCENT,
            border_width=1,
            border_color=CARD_BORDER,
            height=26,
            width=90,
            corner_radius=6,
            command=self._export_log,
        ).pack(side="right")

        card = ctk.CTkFrame(
            self.scroll_container,
            fg_color=CARD_BG,
            corner_radius=12,
            border_width=1,
            border_color=CARD_BORDER,
        )
        card.pack(fill="x", padx=24, pady=(4, 0))

        # Real-time monospace terminal display for audit events
        self.txt_log = ctk.CTkTextbox(
            card,
            height=120,
            font=("Consolas", 11),
            text_color="#38BDF8",
            fg_color="#0A0A0A",
            border_width=1,
            border_color=DIVIDER,
            activate_scrollbars=True,
        )
        self.txt_log.pack(fill="both", padx=12, pady=10)
        self.txt_log.insert("end", "[SYSTEM] Event audit trace ready...\n")
        self.txt_log.configure(state="disabled")

    def _export_log(self) -> None:
        """
        Exports full cumulative session telemetry traces to disk (Section III-D).
        Saves all accepted and suppressed events to mouse_audit.txt for replication.
        """
        # Read complete in-memory session buffer rather than truncated preview queue
        events = (
            self.controller.get_all_session_events()
            if hasattr(self.controller, "get_all_session_events")
            else self.controller.get_latest_events()
        )
        if not events:
            messagebox.showinfo("Export Log", "No events to export yet.")
            return

        default_name = f"mouse_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[
                ("Text Files", "*.txt"),
                ("Log Files", "*.log"),
                ("All Files", "*.*"),
            ],
            title="Save Event Audit Log",
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"=== AI Virtual Mouse Audit Trace ===\n")
                    f.write(
                        f"Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write("=" * 40 + "\n\n")
                    f.write("\n".join(events) + "\n")
                messagebox.showinfo(
                    "Success", f"Log successfully exported to:\n{file_path}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")

    # ── Gesture Guide ────────────────────────────────────────────────

    def _build_gesture_guide(self) -> None:
        """Constructs visual infographic guide mapping gestures to operational states."""
        ctk.CTkLabel(
            self.scroll_container,
            text="✋  Gesture Guide",
            font=("Segoe UI Semibold", 14),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).pack(fill="x", padx=28, pady=(10, 0))

        card = ctk.CTkFrame(
            self.scroll_container,
            fg_color=CARD_BG,
            corner_radius=12,
            border_width=1,
            border_color=CARD_BORDER,
        )
        card.pack(fill="x", padx=24, pady=(4, 0))

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(padx=16, pady=10)

        # Mappings: (Icon, Name, Anatomical Pose Description, Palette Accent)
        gestures = [
            ("☝️", "Move", "Index finger up", GESTURE_COLORS["Move"]),
            ("✌️", "L-Click", "Index + Middle", GESTURE_COLORS["Left_Click"]),
            ("🤟", "R-Click", "Index + Middle + Ring", GESTURE_COLORS["Right_Click"]),
            ("🖐️", "Scroll", "Open palm", GESTURE_COLORS["Scroll"]),
            ("✊", "Drag", "Closed fist", GESTURE_COLORS["Drag"]),
        ]

        for i, (icon, name, desc, color) in enumerate(gestures):
            item = ctk.CTkFrame(inner, fg_color="transparent")
            item.grid(row=0, column=i, padx=10)

            ctk.CTkLabel(item, text=icon, font=("Segoe UI Emoji", 24)).pack()
            ctk.CTkLabel(
                item,
                text=name,
                font=("Segoe UI", 11, "bold"),
                text_color=color,
            ).pack(pady=(2, 0))
            ctk.CTkLabel(
                item,
                text=desc,
                font=("Segoe UI", 9),
                text_color=TEXT_MUTED,
            ).pack()

    # ── Footer ───────────────────────────────────────────────────────

    def _build_footer(self) -> None:
        """Renders subtle author branding and portfolio external hyperlinks."""
        sep = ctk.CTkFrame(self.scroll_container, height=1, fg_color=DIVIDER)
        sep.pack(fill="x", padx=24, pady=(10, 0))

        footer = ctk.CTkFrame(self.scroll_container, fg_color="transparent")
        footer.pack(fill="x", padx=24, pady=(6, 10))

        ctk.CTkLabel(
            footer,
            text="Built by",
            font=("Segoe UI", 11),
            text_color=TEXT_MUTED,
        ).pack(side="left")

        name_btn = ctk.CTkButton(
            footer,
            text="Haithm Al-Akhali",
            font=("Segoe UI Semibold", 12),
            text_color=ACCENT,
            fg_color="transparent",
            hover_color=CARD_BG,
            width=10,
            command=self._open_portfolio,
        )
        name_btn.pack(side="left", padx=(2, 0))

        ctk.CTkLabel(
            footer,
            text="·  Portfolio ↗",
            font=("Segoe UI", 11),
            text_color=TEXT_MUTED,
            cursor="hand2",
        ).pack(side="left")

    # ==================================================================
    # Event Handlers & State Synchronization Callbacks
    # ==================================================================

    def _on_smooth(self, value: int) -> None:
        """Propagates smoothing factor update to the daemon controller (Eq. 4)."""
        if self.controller:
            self.controller.smoothening = value

    def _on_deadzone(self, value: int) -> None:
        """Propagates vertical deadzone update to the daemon controller."""
        if self.controller:
            self.controller.deadzone = value

    def _on_scroll_speed(self, value: int) -> None:
        """Propagates dynamic scroll step multiplier update to the daemon controller."""
        if self.controller:
            self.controller.scroll_speed = value

    def _on_clickDelay(self, value: int) -> None:
        """Propagates refractory interval update (Section III-C) to the daemon controller."""
        if self.controller:
            self.controller.clickDelay = value

    def _toggle(self) -> None:
        """Toggles virtual mouse background daemon execution state."""
        if not self._is_active:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        """Spawns background daemon thread and starts UI preview refresh cycle."""
        self._is_active = True
        self.btn_toggle.configure(
            text="■   Stop Virtual Mouse",
            fg_color=DANGER,
            hover_color=DANGER_HOVER,
        )
        self.status_dot.configure(text_color=SUCCESS)
        self.status_label.configure(text="Running", text_color=SUCCESS)

        # Launch VirtualMouseController on an isolated daemon worker thread
        self.mouse_thread = threading.Thread(
            target=self._run_controller,
            daemon=True,
        )
        self.mouse_thread.start()
        self._start_preview_loop()

    def _stop(self) -> None:
        """Signals daemon thread to terminate and restores idle UI state."""
        self.controller.stop()
        self._is_active = False
        self.btn_toggle.configure(
            text="▶   Start Virtual Mouse",
            fg_color=SUCCESS,
            hover_color=SUCCESS_HOVER,
        )
        self.status_dot.configure(text_color=TEXT_MUTED)
        self.status_label.configure(text="System Ready", text_color=TEXT_SECONDARY)
        self.fps_label.configure(text="FPS: —")
        self.action_label.configure(text="Action: —")
        self._set_preview_placeholder()

    def _run_controller(self) -> None:
        """Worker thread entry-point invoking controller loop."""
        try:
            self.controller.start()
        except RuntimeError as exc:
            logger.error("Controller error: %s", exc)
            self.after(0, lambda: self._show_error(str(exc)))
        except Exception as exc:
            logger.exception("Unexpected controller error")
            self.after(0, lambda: self._show_error(f"Unexpected error: {exc}"))
        finally:
            self.after(0, self._stop)

    def _show_error(self, msg: str) -> None:
        """Renders diagnostic error message onto the UI status bar."""
        self.status_dot.configure(text_color=DANGER)
        self.status_label.configure(text=f"Error: {msg}", text_color=DANGER)

    # ── Camera Preview Refresh Loop ──────────────────────────────────

    def _start_preview_loop(self) -> None:
        """
        Polls the controller thread-safely for newly processed frames and audit logs.
        Refreshes UI widgets at fixed ~33 FPS intervals (Section III-D).
        """
        if not self._is_active:
            return
        frame = self.controller.get_frame()
        if frame is not None:
            self._render_frame(frame)

            # Update live telemetry metrics
            fps = self.controller.current_fps
            action = self.controller.current_action or "—"
            self.fps_label.configure(text=f"FPS: {fps:.0f}")

            color = GESTURE_COLORS.get(action, TEXT_MUTED)
            self.action_label.configure(text=f"Action: {action}", text_color=color)

            # Update live circular event log strip
            if hasattr(self.controller, "get_latest_events"):
                events = self.controller.get_latest_events()
                # Compare the latest entry to suppress unnecessary widget redraws
                if events and getattr(self, "_last_log_entry", None) != events[-1]:
                    self.txt_log.configure(state="normal")
                    self.txt_log.delete("1.0", "end")
                    self.txt_log.insert("end", "\n".join(events) + "\n")
                    self.txt_log.see("end")  # Auto-scroll to latest logged entry
                    self.txt_log.configure(state="disabled")
                    self._last_log_entry = events[-1]

        self._preview_job = self.after(
            self.UPDATE_INTERVAL_MS, self._start_preview_loop
        )

    def _render_frame(self, frame) -> None:
        """Converts raw OpenCV BGR frame into CTkImage for high-DPI canvas rendering."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img = pil_img.resize((self.PREVIEW_W, self.PREVIEW_H), Image.LANCZOS)
        ctk_img = ctk.CTkImage(
            light_image=pil_img,
            dark_image=pil_img,
            size=(self.PREVIEW_W, self.PREVIEW_H),
        )
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label._ctk_image = (
            ctk_img  # Retain reference to prevent garbage collection
        )

    # ── Miscellaneous ────────────────────────────────────────────────

    def _open_portfolio(self) -> None:
        """Launches author portfolio web page in default system browser."""
        webbrowser.open("https://haithm-al-akhali.vercel.app/")

    def _on_close(self) -> None:
        """Handles application window closure and guarantees clean daemon termination."""
        if self._is_active:
            self.controller.stop()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = VirtualMouseApp()
    app.mainloop()
