# settings_window.py
import json
import os
import tkinter as tk
from tkinter import ttk, Pack, filedialog

class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, demo_instance, targets):
        super().__init__(parent)
        self.parent = parent
        self.demo_instance = demo_instance
        self.targets = targets
        self.title("Gesture Settings")
        self.geometry("300x100")

        self.settings_file = "settings.json"

        self.gesture_shortcut_mapping = self.load_settings()

        self.gesture_combobox = ttk.Combobox(self, values=targets, state="readonly")
        self.gesture_combobox.set(targets[0])  # Set default gesture
        self.gesture_combobox.pack(pady=5)

        ###
        # Create a frame for shortcut_entry and browse_button
        entry_frame = ttk.Frame(self)
        entry_frame.pack(pady=5)

        # Pack the shortcut entry with grid
        self.shortcut_entry = ttk.Entry(entry_frame)
        self.shortcut_entry.grid(row=0, column=0, padx=5)
        self.shortcut_entry.configure(width=30)  # Adjust width as needed

        # Create and pack the browse button with grid
        self.browse_button = ttk.Button(entry_frame, text="Browse", command=self.browse_for_path)
        self.browse_button.grid(row=0, column=1, padx=5)

        # Configure the entry_frame to center-align its contents
        entry_frame.grid_columnconfigure(0, weight=1)  # Make the first column (shortcut_entry) expandable
        entry_frame.grid_columnconfigure(1, weight=0)  # Make the second column (browse_button) fixed
        ###
        
        #self.shortcut_entry = ttk.Entry(self)
        #self.shortcut_entry.pack(pady=5)
        #self.shortcut_entry.configure(width=40)        

        # Bind the event to update shortcut entry
        self.gesture_combobox.bind("<<ComboboxSelected>>", lambda event: self.update_shortcut_entry())

        self.update_shortcut_entry()

        # Frame to hold save and delete buttons
        button_frame = ttk.Frame(self)
        button_frame.pack()

        self.browse_button = ttk.Button(self, text="Browse", command=self.browse_for_path)
        self.browse_button.pack(pady=5)

        self.save_button = ttk.Button(button_frame, text="Save", command=self.save_configuration)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.delete_button = ttk.Button(button_frame, text="Delete", command=self.delete_configuration)
        self.delete_button.pack(side=tk.LEFT, padx=5)

    def update_shortcut_entry(self):
        gesture = self.gesture_combobox.get()
        shortcut = self.gesture_shortcut_mapping.get(gesture, "")
        self.shortcut_entry.delete(0, tk.END)
        self.shortcut_entry.insert(0, shortcut)

    def browse_for_path(self):
        selected_path = filedialog.askopenfilename()
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, selected_path)

    def load_settings(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file, "r") as file:
                return json.load(file)
        else:
            return {}

    def save_settings(self, settings):
        with open(self.settings_file, "w") as file:
            json.dump(settings, file)

    def save_configuration(self):
        gesture = self.gesture_combobox.get()
        shortcut = self.shortcut_entry.get()

        # Remove double quotes from paths
        if '"' in shortcut:
            shortcut = shortcut.replace('"', '')

        # Check if the gesture already exists in the mapping
        if gesture in self.gesture_shortcut_mapping:
            # Overwrite the existing shortcut
            self.gesture_shortcut_mapping[gesture] = shortcut
        else:
            # Add a new entry for the gesture
            self.gesture_shortcut_mapping[gesture] = shortcut

        self.demo_instance.set_custom_shortcuts(self.gesture_shortcut_mapping)
        self.save_settings(self.gesture_shortcut_mapping)
        # Trigger the reload of settings in the Demo instance
        self.demo_instance.load_settings()

    def delete_configuration(self):
        gesture = self.gesture_combobox.get()

        # Check if the gesture exists in the mapping
        if gesture in self.gesture_shortcut_mapping:
            # Remove the gesture from the mapping
            del self.gesture_shortcut_mapping[gesture]
            self.demo_instance.set_custom_shortcuts(self.gesture_shortcut_mapping)
            self.save_settings(self.gesture_shortcut_mapping)
            # Trigger the reload of settings in the Demo instance
            self.demo_instance.load_settings()

        # Clear the entry fields
        self.gesture_combobox.set(self.targets[0])
        self.shortcut_entry.delete(0, tk.END)
