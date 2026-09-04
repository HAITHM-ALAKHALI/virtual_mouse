"""
App_GUI — AI Virtual Mouse Control Panel

Premium dark-themed dashboard with an embedded camera preview, live status
indicators, gesture guide, and real-time settings sliders.
"""

import customtkinter as ctk
import threading
import webbrowser
import logging
import cv2
from PIL import Image, ImageTk
from Main import VirtualMouseController

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Appearance ──
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ── Colour Palette ──
BG_DARK = "#111111"
CARD_BG = "#191919"
CARD_BORDER = "#2A2A2A"

ACCENT = "#4587A5"
ACCENT_HOVER = "#4587A57D"

SUCCESS = "#4ADE80"
SUCCESS_HOVER = "#22C55E"

DANGER = "#F87171"
DANGER_HOVER = "#EF4444"

TEXT_PRIMARY = "#F5F5F5"
TEXT_SECONDARY = "#A3A3A3"
TEXT_MUTED = "#737373"

DIVIDER = "#262626"

# Gesture colour coding (matches the OpenCV overlay colours)
GESTURE_COLORS = {
    "Move": "#ff00ff",
    "Left_Click": "#00ff00",
    "Right_Click": "#ff4444",
    "Scroll": "#ffff00",
    "Drag": "#00ffff",
}


class VirtualMouseApp(ctk.CTk):
    """Premium dark-themed control panel for the AI Virtual Mouse."""

    PREVIEW_W = 700
    PREVIEW_H = 500
    UPDATE_INTERVAL_MS = 30  # ~33 FPS GUI refresh

    def __init__(self) -> None:
        super().__init__()

        # ── Window ──
        self.title("AI Virtual Mouse")
        self.geometry("750x780")
        self.minsize(700, 780)
        self.resizable(False, False)
        self.configure(fg_color=BG_DARK)

        # ── Scrollable container ──
        self.scroll_container = ctk.CTkScrollableFrame(
            self,
            fg_color=BG_DARK,
            scrollbar_button_color=CARD_BORDER,
            scrollbar_button_hover_color=TEXT_MUTED,
        )
        self.scroll_container.pack(fill="both", expand=True)

        # ── Controller ──
        self.controller = VirtualMouseController()
        self.mouse_thread = None
        self._is_active = False
        self._preview_job = None

        # ── Build UI ──
        self._build_header()
        self._build_camera_preview()
        self._build_status_bar()
        self._build_settings()
        self._build_gesture_guide()
        self._build_footer()

        # ── Close handler ──
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_header(self) -> None:
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
        self.preview_card = ctk.CTkFrame(
            self.scroll_container,
            fg_color=CARD_BG,
            corner_radius=14,
            border_width=1,
            border_color=CARD_BORDER,
            width=720,  # تحديد عرض ثابت للبطاقة الحاملة
            height=520,
        )
        self.preview_card.pack(fill="none", padx=24, pady=(8, 0))
        self.preview_card.pack_propagate(
            False
        )  # لمنع تمدد البطاقة تلقائياً حسب المحتوى

        # ضبط Label المعاينة في منتصف البطاقة تماماً وبأبعاد الكاميرا الفعلية
        self.preview_label = ctk.CTkLabel(
            self.preview_card,
            text="",
            width=700,
            height=500,
        )
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        self._set_preview_placeholder()

    def _set_preview_placeholder(self) -> None:
        self.preview_label.configure(
            text="📷  Camera preview will appear here\n(Note: The camera may take a while to open)",
            font=("Segoe UI", 14),
            text_color=TEXT_MUTED,
            image=None,
        )

    # ── Status Bar ───────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        bar = ctk.CTkFrame(self.scroll_container, fg_color="transparent")
        bar.pack(fill="x", padx=24, pady=(8, 0))

        # Status dot + label (left)
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

        # FPS + Action (middle-right)
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

        # Start / Stop button (centred below)
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
        # Section title
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

        # Smoothening
        self.lbl_smooth = self._make_slider_row(
            card, "Smoothening", 1, 15, 7, self._on_smooth, row=0
        )
        # Scroll Deadzone
        self.lbl_deadzone = self._make_slider_row(
            card, "Scroll Deadzone", 20, 150, 60, self._on_deadzone, row=1
        )
        # Scroll Speed
        self.lbl_scroll_speed = self._make_slider_row(
            card, "Scroll Speed", 10, 150, 50, self._on_scroll_speed, row=2
        )

    def _make_slider_row(self, parent, label, from_, to, default, command, row):
        """Create a labelled slider row and return the value label."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=16, pady=(6 if row == 0 else 2, 6 if row == 2 else 2))

        name_lbl = ctk.CTkLabel(
            frame,
            text=label,
            font=("Segoe UI", 12),
            text_color=TEXT_SECONDARY,
            width=130,
            anchor="w",
        )
        name_lbl.pack(side="left")

        slider = ctk.CTkSlider(
            frame,
            from_=from_,
            to=to,
            command=command,
            width=320,
            height=16,
            fg_color=DIVIDER,
            progress_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
        )
        slider.set(default)
        slider.pack(side="left", padx=(8, 8))

        val_lbl = ctk.CTkLabel(
            frame,
            text=str(default),
            font=("Segoe UI Semibold", 12),
            text_color=TEXT_PRIMARY,
            width=36,
            anchor="e",
        )
        val_lbl.pack(side="right")

        return val_lbl

    # ── Gesture Guide ────────────────────────────────────────────────

    def _build_gesture_guide(self) -> None:
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

        # (icon, name, description, color)
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
    # Callbacks
    # ==================================================================

    def _on_smooth(self, value) -> None:
        v = int(value)
        self.lbl_smooth.configure(text=str(v))
        self.controller.smoothening = v

    def _on_deadzone(self, value) -> None:
        v = int(value)
        self.lbl_deadzone.configure(text=str(v))
        self.controller.deadzone = v

    def _on_scroll_speed(self, value) -> None:
        v = int(value)
        self.lbl_scroll_speed.configure(text=str(v))
        self.controller.scroll_speed = v

    def _toggle(self) -> None:
        if not self._is_active:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        self._is_active = True
        self.btn_toggle.configure(
            text="■   Stop Virtual Mouse",
            fg_color=DANGER,
            hover_color=DANGER_HOVER,
        )
        self.status_dot.configure(text_color=SUCCESS)
        self.status_label.configure(text="Running", text_color=SUCCESS)

        self.mouse_thread = threading.Thread(
            target=self._run_controller,
            daemon=True,
        )
        self.mouse_thread.start()
        self._start_preview_loop()

    def _stop(self) -> None:
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
        """Target for the worker thread."""
        try:
            self.controller.start()
        except RuntimeError as exc:
            logger.error("Controller error: %s", exc)
            # Schedule UI update on main thread
            self.after(0, lambda: self._show_error(str(exc)))
        except Exception as exc:
            logger.exception("Unexpected controller error")
            self.after(0, lambda: self._show_error(f"Unexpected error: {exc}"))
        finally:
            self.after(0, self._stop)

    def _show_error(self, msg: str) -> None:
        """Flash an error message on the status bar."""
        self.status_dot.configure(text_color=DANGER)
        self.status_label.configure(text=f"Error: {msg}", text_color=DANGER)

    # ── Camera Preview Loop ──────────────────────────────────────────

    def _start_preview_loop(self) -> None:
        if not self._is_active:
            return
        frame = self.controller.get_frame()
        if frame is not None:
            self._render_frame(frame)

            # Update stats
            fps = self.controller.current_fps
            action = self.controller.current_action or "—"
            self.fps_label.configure(text=f"FPS: {fps:.0f}")

            color = GESTURE_COLORS.get(action, TEXT_MUTED)
            self.action_label.configure(text=f"Action: {action}", text_color=color)

        self._preview_job = self.after(
            self.UPDATE_INTERVAL_MS, self._start_preview_loop
        )

    def _render_frame(self, frame) -> None:
        """Convert an OpenCV BGR frame to a CTk-compatible image and display it."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img = pil_img.resize((self.PREVIEW_W, self.PREVIEW_H), Image.LANCZOS)
        ctk_img = ctk.CTkImage(
            light_image=pil_img,
            dark_image=pil_img,
            size=(self.PREVIEW_W, self.PREVIEW_H),
        )
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label._ctk_image = ctk_img  # prevent GC

    # ── Misc ─────────────────────────────────────────────────────────

    def _open_portfolio(self) -> None:
        webbrowser.open("https://haithm-al-akhali.vercel.app/")

    def _on_close(self) -> None:
        """Graceful shutdown on window close."""
        if self._is_active:
            self.controller.stop()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = VirtualMouseApp()
    app.mainloop()
