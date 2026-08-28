import customtkinter as ctk
import threading
import webbrowser
from Main import run_virtual_mouse, stop_virtual_mouse, set_smoothening, set_deadzone

# إعدادات المظهر العام
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class VirtualMouseApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # إعدادات النافذة
        self.title("AI Virtual Mouse Control Panel")
        self.geometry("400x460")
        self.resizable(False, False)

        # العنوان الرئيسي
        self.label_title = ctk.CTkLabel(
            self, text="إعدادات الماوس الافتراضي", font=("Arial", 20, "bold")
        )
        self.label_title.pack(pady=(20, 15))

        # 1. شريط التحكم في التنعيم (Smoothening)
        self.label_smooth = ctk.CTkLabel(
            self, text="مستوى التنعيم (Smoothening): 7", font=("Arial", 13)
        )
        self.label_smooth.pack(pady=(5, 0))
        self.slider_smooth = ctk.CTkSlider(
            self, from_=1, to=15, command=self.update_smooth
        )
        self.slider_smooth.set(7)
        self.slider_smooth.pack(pady=8)

        # 2. شريط التحكم في التمرير (Scroll Deadzone)
        self.label_scroll = ctk.CTkLabel(
            self, text="منطقة أمان التمرير (Deadzone): 60", font=("Arial", 13)
        )
        self.label_scroll.pack(pady=(15, 0))
        self.slider_scroll = ctk.CTkSlider(
            self, from_=20, to=150, command=self.update_scroll
        )
        self.slider_scroll.set(60)
        self.slider_scroll.pack(pady=8)

        # 3. زر التشغيل والإيقاف
        self.btn_start = ctk.CTkButton(
            self,
            text="تشغيل الكاميرا والماوس",
            font=("Arial", 14, "bold"),
            fg_color="green",
            hover_color="darkgreen",
            height=40,
            command=self.start_mouse,
        )
        self.btn_start.pack(pady=20)

        # فاصل مرئي تفصيلي
        self.separator = ctk.CTkFrame(self, height=2, fg_color="#333333")
        self.separator.pack(fill="x", padx=30, pady=5)

        # معلومات المطور (Developer Info)
        self.label_dev_title = ctk.CTkLabel(
            self,
            text="Developer Info",
            font=("Arial", 13, "bold"),
            text_color="#aaaaaa",
        )
        self.label_dev_title.pack(pady=(5, 0))

        self.label_dev_name = ctk.CTkLabel(
            self,
            text="Haithm Al-Akhali",
            font=("Arial", 15, "bold"),
            text_color="#1f6aa5",
        )
        self.label_dev_name.pack(pady=2)

        # زر التوجيه للموقع الشخصي (Portfolio Button)
        self.btn_portfolio = ctk.CTkButton(
            self,
            text="زيارة موقعي الشخصي",
            font=("Arial", 12, "bold"),
            fg_color="#1f6aa5",
            hover_color="#144870",
            height=32,
            command=self.open_portfolio,
        )
        self.btn_portfolio.pack(pady=(5, 15))

        self.is_running = False

    def update_smooth(self, value):
        self.label_smooth.configure(text=f"مستوى التنعيم (Smoothening): {int(value)}")
        set_smoothening(value)

    def update_scroll(self, value):
        self.label_scroll.configure(text=f"منطقة أمان التمرير (Deadzone): {int(value)}")
        set_deadzone(value)

    def open_portfolio(self):
        # فتح الرابط مباشرة في المتصفح الافتراضي
        webbrowser.open("https://haithm-al-akhali.vercel.app/")

    def start_mouse(self):
        if not self.is_running:
            self.is_running = True
            self.btn_start.configure(
                text="إيقاف الماوس", fg_color="red", hover_color="darkred"
            )

            # تشغيل الماوس في Thread منفصل لمنع تجميد الواجهة
            self.mouse_thread = threading.Thread(target=run_virtual_mouse, daemon=True)
            self.mouse_thread.start()
        else:
            self.is_running = False
            self.btn_start.configure(
                text="تشغيل الكاميرا والماوس", fg_color="green", hover_color="darkgreen"
            )
            stop_virtual_mouse()


if __name__ == "__main__":
    app = VirtualMouseApp()
    app.mainloop()
