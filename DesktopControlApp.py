import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from demo import Demo, parse_arguments, build_model
import argparse
from omegaconf import OmegaConf
from typing import Optional, Tuple
import pyautogui

from settings_window import SettingsWindow

class DesktopControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Desktop Control App")

        # Initialize the demo instance
        self.demo = Demo()

        # Create a settings window
        self.settings_window = None

        # Create a button to start gesture recognition
        self.start_button = ttk.Button(root, text="Start Gesture Recognition", command=self.start_gesture_recognition)
        self.start_button.pack()

        # Create a button to open the settings window
        self.settings_button = ttk.Button(root, text="Open Settings", command=self.open_settings)
        self.settings_button.pack()

        # Create a label to display the camera feed
        self.camera_label = ttk.Label(root)
        self.camera_label.pack()

    def open_settings(self):
        if self.settings_window:
            self.settings_window.destroy()
        self.settings_window = SettingsWindow(self.root, self.demo, ["call", "four", "mute", "ok", "palm", "stop", "stop inverted", "two up", "two up inverted", "three2", "peace inverted"])

    def start_gesture_recognition(self):
        # Hide the main window
        self.root.iconify()
        # Start the gesture recognition logic from demo.py
        args = parse_arguments()
        conf = OmegaConf.load(args.path_to_config)
        model = build_model(
            model_name=conf.model.name,
            num_classes=len(conf.dataset.targets) + 1,
            checkpoint=conf.model.get("checkpoint", None),
            device=conf.device,
            pretrained=conf.model.pretrained,
        )

        model.eval()
        self.demo.run(model, num_hands=100, threshold=0.8, landmarks=args.landmarks)
        # Show the main window when gesture recognition is finished
        self.root.deiconify()

def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo detection...")

    # Provide default values for the required arguments
    default_config_path = "D:\\pwr\\praca dyplomowa\\GestureRecognition\\PracaDyplomowaProjekt\\default.yaml"
    default_landmarks = True

    parser.add_argument("-p", "--path_to_config", default=default_config_path, type=str, help="Path to config")
    parser.add_argument("-lm", "--landmarks", default=default_landmarks, action="store_true", help="Use landmarks")

    known_args, _ = parser.parse_known_args(params)
    return known_args

if __name__ == "__main__":
    root = tk.Tk()
    app = DesktopControlApp(root)
    root.mainloop()
