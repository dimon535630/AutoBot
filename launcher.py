import json
import tkinter as tk
from datetime import timedelta
from tkinter import messagebox, ttk

import keyboard

import main
from license_manager import LicenseManager

CONFIG_PATH = "config.json"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class LicensePanel(ttk.LabelFrame):
    def __init__(self, parent, license_manager: LicenseManager, on_status_change):
        super().__init__(parent, text="Лицензирование", padding=12, style="Card.TLabelframe")
        self.license_manager = license_manager
        self.on_status_change = on_status_change

        self.key_to_activate = tk.StringVar()
        self.status_var = tk.StringVar(value="Лицензия не активна")

        ttk.Label(self, textvariable=self.status_var, style="Card.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )

        ttk.Label(self, text="Ввести ключ:", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(self, textvariable=self.key_to_activate, width=30).grid(row=1, column=1, sticky="ew", pady=(10, 0))
        ttk.Button(self, text="Активировать", command=self.activate_key, style="Accent.TButton").grid(
            row=1, column=2, padx=8, pady=(10, 0)
        )

        ttk.Button(self, text="Сбросить лицензию", command=self.deactivate_key).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

        ttk.Label(
            self,
            text="Произойдет привязка пк.",
            style="Hint.TLabel",
            wraplength=500,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.columnconfigure(1, weight=1)

    def refresh_status(self):
        status = self.license_manager.get_status()
        if status.is_active and status.expires_at:
            left = timedelta(seconds=status.seconds_left)
            self.status_var.set(f"Активен ключ: {status.key_value} | осталось: {left}")
        else:
            self.status_var.set("Лицензия не активна")
        self.on_status_change(status.is_active)

    def activate_key(self):
        try:
            self.license_manager.activate_with_key(self.key_to_activate.get())
            self.key_to_activate.set("")
            self.refresh_status()
            messagebox.showinfo("ОК", "Лицензия активирована")
        except Exception as e:
            messagebox.showerror("Ошибка активации", str(e))

    def deactivate_key(self):
        self.license_manager.deactivate()
        self.refresh_status()


class LicenseActivationWindow(tk.Tk):
    def __init__(self, license_manager: LicenseManager):
        super().__init__()
        self.title("Активация лицензии")
        self.geometry("560x280")
        self.minsize(520, 260)

        self.license_manager = license_manager
        self.activated = False

        self._configure_styles()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_styles(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        bg_main = "#0f131c"
        bg_card = "#171e2a"
        fg_primary = "#e7edf9"
        fg_secondary = "#a8b3c6"

        self.configure(bg=bg_main)

        style.configure("Main.TFrame", background=bg_main)
        style.configure("Card.TFrame", background=bg_card)
        style.configure("Header.TLabel", background=bg_main, foreground=fg_primary, font=("Segoe UI", 16, "bold"))
        style.configure("SubHeader.TLabel", background=bg_main, foreground=fg_secondary, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=bg_card, foreground=fg_primary)
        style.configure("Hint.TLabel", background=bg_card, foreground=fg_secondary)
        style.configure("Card.TLabelframe", background=bg_card, foreground=fg_primary)
        style.configure("Card.TLabelframe.Label", background=bg_card, foreground=fg_primary)
        style.configure("Accent.TButton", background="#4f8cff", foreground="white", padding=8)
        style.map("Accent.TButton", background=[("active", "#6ba0ff")])

    def _build_ui(self):
        root = ttk.Frame(self, padding=14, style="Main.TFrame")
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="FishingBot", style="Header.TLabel").pack(anchor="w")
        ttk.Label(root, text="Перед началом работы активируйте ключ", style="SubHeader.TLabel").pack(
            anchor="w", pady=(2, 10)
        )

        self.license_panel = LicensePanel(root, self.license_manager, self.on_license_change)
        self.license_panel.pack(fill="both", expand=True)
        self.license_panel.refresh_status()

    def on_license_change(self, active: bool):
        if active:
            self.activated = True
            messagebox.showinfo("ОК", "Лицензия активирована. Открываем основное окно.")
            self.destroy()

    def on_close(self):
        self.activated = False
        self.destroy()


class Launcher(tk.Tk):
    CYCLE_RESET_LIMIT = 7

    def __init__(self, license_manager: LicenseManager):
        super().__init__()
        self.title("Рыболовный помощник")
        self.geometry("560x360")
        self.minsize(540, 340)

        self._hotkey_ids = []
        self.ctl = main.BotController()
        self.license_manager = license_manager
        self.cfg = {}

        self.status_var = tk.StringVar(value="STOPPED")
        self.license_info_var = tk.StringVar(value="Лицензия: проверка...")
        self.reset_enabled_var = tk.BooleanVar(value=True)

        self._configure_styles()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.reload_config()
        self.sync_reset_options()

        self.after(200, self.poll_status)
        self.after(1000, self.poll_license)


    def _configure_styles(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        bg_main = "#0f131c"
        bg_card = "#171e2a"
        fg_primary = "#e7edf9"
        fg_secondary = "#a8b3c6"

        self.configure(bg=bg_main)

        style.configure("Main.TFrame", background=bg_main)
        style.configure("Card.TFrame", background=bg_card)

        style.configure("Header.TLabel", background=bg_main, foreground=fg_primary, font=("Segoe UI", 18, "bold"))
        style.configure("SubHeader.TLabel", background=bg_main, foreground=fg_secondary, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background="#223149", foreground="#dce7ff", font=("Segoe UI", 10, "bold"))

        style.configure("Card.TLabel", background=bg_card, foreground=fg_primary)
        style.configure("Hint.TLabel", background=bg_card, foreground=fg_secondary)

        style.configure("Card.TLabelframe", background=bg_card, foreground=fg_primary)
        style.configure("Card.TLabelframe.Label", background=bg_card, foreground=fg_primary)

        style.configure("TButton", padding=8)
        style.configure("Accent.TButton", background="#4f8cff", foreground="white", padding=8)
        style.map("Accent.TButton", background=[("active", "#6ba0ff")])

        style.configure("TCheckbutton", background=bg_card, foreground=fg_primary)
        style.map("TCheckbutton", background=[("active", bg_card)])

    def _build_ui(self):
        root = ttk.Frame(self, padding=14, style="Main.TFrame")
        root.pack(fill="both", expand=True)
        self.content_root = root

        header = ttk.Frame(root, style="Main.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="FishingBot", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="Управление рыбалкой в один клик", style="SubHeader.TLabel").pack(side="left", padx=12)
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel", padding=(12, 6)).pack(side="right")

        ttk.Label(root, textvariable=self.license_info_var, style="SubHeader.TLabel").pack(anchor="w", pady=(6, 0))

        control_card = ttk.Frame(root, style="Card.TFrame", padding=12)
        control_card.pack(fill="x", pady=(12, 8))

        controls_row = ttk.Frame(control_card, style="Card.TFrame")
        controls_row.pack(fill="x")
        ttk.Button(controls_row, text="START", command=self.on_start, style="Accent.TButton").pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )
        ttk.Button(controls_row, text="STOP", command=self.on_stop).pack(side="left", fill="x", expand=True, padx=(5, 0))

        modes = ttk.Frame(control_card, style="Card.TFrame")
        modes.pack(fill="x", pady=(10, 0))
        ttk.Button(modes, text="Забрать себе", command=self.ctl.set_take_mode).pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        ttk.Button(modes, text="Отпустить", command=self.ctl.set_release_mode).pack(
            side="left", fill="x", expand=True, padx=(4, 0)
        )

        reset_box = ttk.LabelFrame(root, text="Сброс", padding=12, style="Card.TLabelframe")
        reset_box.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(
            reset_box,
            text=f"Включить замену наживки после {self.CYCLE_RESET_LIMIT} циклов",
            variable=self.reset_enabled_var,
            command=self.sync_reset_options,
        ).pack(anchor="w")

        ttk.Button(root, text="Reload config.json", command=self.on_reload).pack(fill="x", pady=(0, 10))

    def apply_config_to_bot(self, cfg: dict):
        sound = cfg.get("sound", {})
        if "file" in sound:
            main.sound_file_path = sound["file"]
        if "enabled" in sound:
            main.SOUND_ENABLED = bool(sound["enabled"])

        behavior = cfg.get("behavior", {})
        reward_action = behavior.get("reward_action", "take")
        if reward_action == "release":
            self.ctl.set_release_mode()
        else:
            self.ctl.set_take_mode()

    def setup_hotkeys(self, cfg: dict):
        self._clear_hotkeys()
        hk = cfg.get("hotkeys", {})
        self._safe_add_hotkey(hk.get("start", "+"), self.on_start)
        self._safe_add_hotkey(hk.get("stop", "-"), self.on_stop)
        self._safe_add_hotkey(hk.get("press_esc", "0"), self.ctl.press_esc)

    def _safe_add_hotkey(self, hotkey, callback):
        try:
            hk_id = keyboard.add_hotkey(hotkey, callback)
            self._hotkey_ids.append(hk_id)
        except Exception as e:
            print(f"[WARN] hotkey {hotkey} disabled: {e}")

    def _clear_hotkeys(self):
        for hk_id in self._hotkey_ids:
            try:
                keyboard.remove_hotkey(hk_id)
            except Exception:
                pass
        self._hotkey_ids = []

    def reload_config(self):
        try:
            self.cfg = load_config()
            self.apply_config_to_bot(self.cfg)
            self.setup_hotkeys(self.cfg)
        except Exception as e:
            messagebox.showerror("Config error", str(e))

    def sync_reset_options(self):
        self.ctl.bot.set_post_cycle_reset(self.reset_enabled_var.get())
        self.ctl.bot.set_cycle_limit(self.CYCLE_RESET_LIMIT)

    def on_start(self):
        if not self.license_manager.get_status().is_active:
            messagebox.showwarning("Лицензия", "Сначала активируйте ключ доступа")
            return
        self.ctl.start()

    def on_stop(self):
        self.ctl.stop()

    def on_reload(self):
        self.reload_config()
        messagebox.showinfo("OK", "config.json перезагружен")

    def poll_status(self):
        self.status_var.set("RUNNING" if self.ctl.bot.bot_running else "STOPPED")
        self.after(200, self.poll_status)

    def poll_license(self):
        status = self.license_manager.get_status()
        if status.is_active and status.expires_at:
            left = timedelta(seconds=status.seconds_left)
            self.license_info_var.set(f"Лицензия активна, осталось: {left}")
        else:
            self.license_info_var.set("Лицензия неактивна")

        if not status.is_active:
            self.on_stop()
            messagebox.showwarning("Лицензия", "Лицензия неактивна. Приложение будет закрыто.")
            self.on_close()
            return
        self.after(1000, self.poll_license)

    def on_close(self):
        try:
            self.ctl.stop()
        finally:
            self._clear_hotkeys()
            self.destroy()


if __name__ == "__main__":
    manager = LicenseManager()
    status = manager.get_status()

    if not status.is_active:
        activation = LicenseActivationWindow(manager)
        activation.mainloop()
        if not activation.activated:
            raise SystemExit

    app = Launcher(manager)
    app.mainloop()
