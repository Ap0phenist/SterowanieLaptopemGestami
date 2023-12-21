import argparse
import logging
import time
from typing import Optional, Tuple
import sys
print(sys.executable)
import json

import cv2
import mediapipe as mp
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms import functional as f
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from constants import targets
from detector.models.model import TorchVisionModel
from detector.utils import build_model

import pyautogui
import keyboard

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Demo:
    detected_gestures = []
    last_detected_gesture = None
    is_muted = False
    custom_shortcuts = {}  # Store custom shortcuts
    gestures_list = ["call", "four", "mute", "ok", "palm", "stop", "two up inverted", "three2", "peace inverted"]
    gesture_recognition_enabled = True  # Track the state of gesture recognizing
    

    def __init__(self):        
        # Load the configuration file and set custom shortcuts
        self.load_settings()

    def load_settings(self):
        config_file_path = "settings.json"  # Adjust the file path based on your project structure
        try:
            with open(config_file_path, "r") as f:
                settings = json.load(f)
                self.custom_shortcuts = settings  # Load gestures directly into custom_shortcuts
                print(self.custom_shortcuts)
        except FileNotFoundError:
            print(f"Settings file not found at {config_file_path}")

    #@classmethod
    #def switch_gesture_recognition(self):
    #    if self.gesture_recognition_enabled == True:
    #        self.gesture_recognition_enabled = False
    #    else:
    #        self.gesture_recognition_enabled = True

    @classmethod
    def switch_gesture_recognition(cls):
        cls.gesture_recognition_enabled = not cls.gesture_recognition_enabled

    def execute_custom_shortcut(self, gesture):
        shortcut_or_path = self.custom_shortcuts.get(gesture)
        
        if shortcut_or_path:
            if os.path.exists(shortcut_or_path):  # Check if it's a valid path
                os.startfile(shortcut_or_path)  # Execute the .exe file
            else:
                pyautogui.hotkey(*shortcut_or_path.split('+'))  # Execute keyboard shortcut
    def switch_to_next_tab(num_times=1):
        for _ in range(num_times):
            pyautogui.hotkey('ctrl', 'tab')

    def minimize_all_windows():
        pyautogui.hotkey('winleft', 'm')

    def set_custom_shortcuts(self, shortcut_mapping):
        self.custom_shortcuts = shortcut_mapping

    def preprocess(img: np.ndarray) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        width, height = image.size

        image = ImageOps.pad(image, (max(width, height), max(width, height)))
        padded_width, padded_height = image.size
        image = image.resize((320, 320))

        img_tensor = f.pil_to_tensor(image)
        img_tensor = f.convert_image_dtype(img_tensor)
        img_tensor = img_tensor[None, :, :, :]
        return img_tensor, (width, height), (padded_width, padded_height)

    def run(self, detector: TorchVisionModel, num_hands: int = 2, threshold: float = 0.5, landmarks: bool = False) -> None:
        """
        Run detection model and draw bounding boxes on frame
        Parameters
        ----------
        detector : TorchVisionModel
            Detection model
        num_hands:
            Min hands to detect
        threshold : float
            Confidence threshold
        landmarks : bool
            Detect landmarks
        """

        if landmarks:
            hands = mp.solutions.hands.Hands(
                model_complexity=0, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8
            )

        cap = cv2.VideoCapture(0)
        # Set the window name
        cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
        # Resize the window
        cv2.resizeWindow("Gesture Recognition", 320, 240)
        # Move the window to the top-left corner
        cv2.moveWindow("Gesture Recognition", 0, 0)

        print("Window created and configured.")

        t1 = cnt = 0
        while cap.isOpened():
            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if ret:
                processed_frame, size, padded_size = Demo.preprocess(frame)
                with torch.no_grad():
                    output = detector(processed_frame)[0]
                boxes = output["boxes"][:num_hands]
                scores = output["scores"][:num_hands]
                labels = output["labels"][:num_hands]
                if landmarks:
                    results = hands.process(frame[:, :, ::-1])
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                mp_drawing_styles.DrawingSpec(color=[0, 255, 0], thickness=2, circle_radius=1),
                                mp_drawing_styles.DrawingSpec(color=[255, 255, 255], thickness=1, circle_radius=1),
                            )

                for i in range(min(num_hands, len(boxes))):
                    if scores[i] > threshold:
                        width, height = size
                        padded_width, padded_height = padded_size
                        scale = max(width, height) / 320

                        padding_w = abs(padded_width - width) // (2 * scale)
                        padding_h = abs(padded_height - height) // (2 * scale)

                        x1 = int((boxes[i][0] - padding_w) * scale)
                        y1 = int((boxes[i][1] - padding_h) * scale)
                        x2 = int((boxes[i][2] - padding_w) * scale)
                        y2 = int((boxes[i][3] - padding_h) * scale)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, thickness=3)
                        cv2.putText(
                            frame,
                            targets[int(labels[i])],
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            thickness=3,
                        )

                        # Store the detected gesture in the list
                        detected_gesture = targets[int(labels[i])]
                        Demo.detected_gestures.append(detected_gesture)

                        # Print the gesture only if it has changed
                        current_gesture = Demo.detected_gestures[-1] if Demo.detected_gestures else None
                        if current_gesture != Demo.last_detected_gesture:
                            print("Detected Gesture:", current_gesture)
                            Demo.last_detected_gesture = current_gesture
                            # Enable/Disable gesture recognition
                            if current_gesture == "two up inverted":
                                Demo.switch_gesture_recognition()
                                print('gesture recognition: ', Demo.gesture_recognition_enabled)
                            if Demo.gesture_recognition_enabled:
                                # Mute or unmute based on the recognized gesture
                                if current_gesture == "dislike" and not Demo.is_muted:
                                    pyautogui.press("volumemute")  # Adjust this based on your system
                                    Demo.is_muted = True
                                elif current_gesture == "like" and Demo.is_muted:
                                    pyautogui.press("volumemute")  # Adjust this based on your system
                                    Demo.is_muted = False
                                # Switch to the next tab in Google Chrome
                                elif current_gesture == "one":                           
                                    Demo.switch_to_next_tab()
                                elif current_gesture == "peace":
                                    Demo.switch_to_next_tab(num_times=2)
                                elif current_gesture == "three":
                                    Demo.switch_to_next_tab(num_times=3)
                                # Minimize all windows
                                elif current_gesture == "fist":                                
                                    Demo.minimize_all_windows()
                                # Open Start menu
                                elif current_gesture == "two up":                                
                                    pyautogui.hotkey('ctrl', 'esc')
                                # Close active window
                                elif current_gesture == "stop inverted":
                                    print ("closing")
                                    pyautogui.hotkey('alt', 'f4')
                                # Open file explorer
                                elif current_gesture == "stop":
                                    pyautogui.hotkey('win', 'e')
                                # Execute custom shortcut for the recognized gesture
                                elif current_gesture in self.custom_shortcuts:
                                    print("custom shortcut")
                                    Demo().execute_custom_shortcut(gesture=current_gesture)
                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {cnt}", (30, 30), FONT, 1, COLOR, 2)
                cnt += 1

                # Display the frame in the "Gesture Recognition" window
                cv2.imshow("Gesture Recognition", frame)

                key = cv2.waitKey(1)
                if key == ord("q") or cv2.getWindowProperty("Gesture Recognition", cv2.WND_PROP_VISIBLE) < 1:
                    print("Exiting loop.")
                    break
        # Release the video capture and close the window
        cap.release()
        cv2.destroyAllWindows()


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo detection...")

    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")

    parser.add_argument("-lm", "--landmarks", required=False, action="store_true", help="Use landmarks")

    known_args, _ = parser.parse_known_args(params)
    return known_args

def run_demo(self, args):
        # Start the gesture recognition logic
        conf = OmegaConf.load(args.path_to_config)
        model = build_model(
            model_name=conf.model.name,
            num_classes=len(conf.dataset.targets) + 1,
            checkpoint=conf.model.get("checkpoint", None),
            device=conf.device,
            pretrained=conf.model.pretrained,
        )

        model.eval()
        self.run(model, num_hands=100, threshold=0.8, landmarks=args.landmarks)


if __name__ == "__main__":
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
    if model is not None:
        Demo.run(model, num_hands=100, threshold=0.8, landmarks=args.landmarks)
