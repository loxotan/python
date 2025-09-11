import sys
import os
import traceback
import copy

# --- Debug Logging to File ---
log_file_path = 'app_debug.log' # Log file will be created next to the executable
# In a frozen app, the executable's directory is the current working directory.
if getattr(sys, 'frozen', False):
    log_file_path = os.path.join(os.path.dirname(sys.executable), 'app_debug.log')

try:
    # Clear log file on start
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    class Logger(object):
        def __init__(self, filename="default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8')

        def write(self, message):
            if self.terminal:
                self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    sys.stdout = Logger(log_file_path)
    sys.stderr = sys.stdout # Redirect stderr to the same logger

    print("---Application Start---")

except Exception as e:
    # If logging setup fails, we can't do much but this might show up in console
    print(f"Failed to set up file logging: {e}")
    traceback.print_exc()
# --- End Debug Logging ---

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog
import shutil
import re
from datetime import datetime
import difflib
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
from torchvision import models, transforms
import PIL.Image, PIL.ImageTk
import json
import glob


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, current_structure, current_file_format, save_callback):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("600x600")
        self.transient(parent)
        self.grab_set()

        self.save_callback = save_callback
        self.structure = [item['format'] for item in current_structure]
        self.file_format = current_file_format
        self.entries = []

        main_frame = tk.Frame(self)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # --- Folder Structure ---
        folder_frame = tk.LabelFrame(main_frame, text="Folder Structure", padx=5, pady=5)
        folder_frame.pack(pady=5, fill="x")
        tk.Label(folder_frame, text="Configure the folder hierarchy from top to bottom.").pack(anchor="w")
        self.levels_frame = tk.Frame(folder_frame)
        self.levels_frame.pack(pady=5, fill="x")
        self.draw_levels()
        add_button = tk.Button(folder_frame, text="Add Level", command=self.add_level)
        add_button.pack(pady=5)

        # --- File Naming ---
        file_frame = tk.LabelFrame(main_frame, text="File Naming (extension is kept automatically)", padx=5, pady=5)
        file_frame.pack(pady=5, fill="x")
        tk.Label(file_frame, text="Configure the format for new file names.").pack(anchor="w")
        self.file_format_entry = tk.Entry(file_frame, width=80)
        self.file_format_entry.insert(0, self.file_format)
        self.file_format_entry.pack(side=tk.LEFT, padx=5, expand=True, fill="x")

        # --- Placeholders ---
        placeholder_frame = tk.LabelFrame(main_frame, text="Available Placeholders", padx=5, pady=5)
        placeholder_frame.pack(pady=5, fill="both", expand=True)
        placeholder_text = """--- Date & Time (from photo's creation date) ---
  %Y: Year (2023), %y: Year (23), %m: Month (09), %d: Day (05)
  %H: Hour (14), %M: Minute (30), %S: Second (05)
  %A: Weekday (Wednesday), %a: Weekday (Wed)

--- Special Placeholders (for Folders & Files) ---
  {weekday}: Weekday in Korean (수)
  {organizer_name}: The 'Organizer Name' from the main window
  {patient_folder_name}: Patient's folder name (e.g., 홍길동 1234567)

--- File-specific (for File Naming only) ---
  {original_name}: The original name of the file without extension
  {counter}: A sequential number for this session (e.g., 001)"""
        tk.Label(placeholder_frame, text=placeholder_text, justify=tk.LEFT).pack(pady=2, anchor="w")

        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=10)

    def draw_levels(self):
        for widget in self.levels_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        for i, fmt in enumerate(self.structure):
            level_frame = tk.Frame(self.levels_frame)
            level_frame.pack(fill="x", pady=2)

            tk.Label(level_frame, text=f"Level {i+1}:").pack(side=tk.LEFT)
            entry = tk.Entry(level_frame, width=50)
            entry.insert(0, fmt)
            entry.pack(side=tk.LEFT, padx=5, expand=True, fill="x")
            self.entries.append(entry)

            up_button = tk.Button(level_frame, text="▲", command=lambda i=i: self.move_level(i, -1))
            up_button.pack(side=tk.LEFT)
            if i == 0: up_button.config(state="disabled")

            down_button = tk.Button(level_frame, text="▼", command=lambda i=i: self.move_level(i, 1))
            down_button.pack(side=tk.LEFT)
            if i == len(self.structure) - 1: down_button.config(state="disabled")

            remove_button = tk.Button(level_frame, text="Remove", command=lambda i=i: self.remove_level(i))
            remove_button.pack(side=tk.LEFT, padx=5)

    def add_level(self):
        self.structure.append("new_level_format")
        self.draw_levels()

    def remove_level(self, index):
        if len(self.structure) > 1:
            self.structure.pop(index)
            self.draw_levels()
        else:
            messagebox.showwarning("Warning", "At least one level is required.", parent=self)

    def move_level(self, index, direction):
        if 0 <= index + direction < len(self.structure):
            self.structure.insert(index + direction, self.structure.pop(index))
            self.draw_levels()

    def save(self):
        new_structure_formats = [entry.get().strip() for entry in self.entries]
        if any(not fmt for fmt in new_structure_formats):
            messagebox.showerror("Error", "Folder format cannot be empty.", parent=self)
            return
        new_structure_config = [{'format': fmt} for fmt in new_structure_formats]

        new_file_format = self.file_format_entry.get().strip()
        if not new_file_format:
            messagebox.showerror("Error", "File naming format cannot be empty.", parent=self)
            return

        self.save_callback(new_structure_config, new_file_format)
        self.destroy()


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cpu")

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
PATHS_FILE = os.path.join(application_path, 'paths.json')

def build_model(num_classes):
    model = models.mobilenet_v2(weights=None, progress=False)
    mobilenet_weights_path = resource_path('mobilenet_v2-7ebf99e0.pth')
    if os.path.exists(mobilenet_weights_path):
        state_dict = torch.load(mobilenet_weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"MobileNetV2 weights file not found at {mobilenet_weights_path}.")
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 1))
    return model.to(DEVICE)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_korean_weekday(weekday):
    return ['월', '화', '수', '목', '금', '토', '일'][weekday]

class PhotoOrganizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Organizer v2.7.1")
        self.root.geometry("800x600+100+100")
        self.root.resizable(False, False)

        self.is_organizing = False
        self.model = None
        self.patient_info_cache = None
        self.patient_db_lock = threading.Lock()
        self.all_settings = {}
        self.folder_structure_config = []
        self.file_naming_format = ""
        self.last_organizing_timestamp = 0.0
        self.last_sync_timestamp = 0.0

        # --- Profile Management UI ---
        profile_frame = tk.LabelFrame(root, text="Preset Management", padx=5, pady=5)
        profile_frame.pack(padx=10, pady=5, fill="x")

        tk.Label(profile_frame, text="Preset:").pack(side=tk.LEFT, padx=(0, 5))
        self.profile_combobox = ttk.Combobox(profile_frame, width=30, state="readonly")
        self.profile_combobox.pack(side=tk.LEFT, padx=5)
        self.profile_combobox.bind("<<ComboboxSelected>>", self.on_profile_select)

        self.save_profile_button = tk.Button(profile_frame, text="Save CurrentPreset", command=self.save_current_profile)
        self.save_profile_button.pack(side=tk.LEFT, padx=5)
        self.new_profile_button = tk.Button(profile_frame, text="New Preset", command=self.new_profile)
        self.new_profile_button.pack(side=tk.LEFT, padx=5)
        self.delete_profile_button = tk.Button(profile_frame, text="Delete Preset", command=self.delete_profile)
        self.delete_profile_button.pack(side=tk.LEFT, padx=5)

        # --- Main UI Frames ---
        self.folder_frame = tk.Frame(root, padx=5, pady=2)
        self.folder_frame.pack(padx=5, pady=2)

        # Row 0: Source Directory
        self.source_dir_label = tk.Label(self.folder_frame, text="Source Directory:")
        self.source_dir_entry = tk.Entry(self.folder_frame, width=50)
        self.source_dir_button = tk.Button(self.folder_frame, text="Browse...", command=self.browse_source_dir)
        self.source_dir_label.grid(row=0, column=0, padx=5, pady=1, sticky="w")
        self.source_dir_entry.grid(row=0, column=1, padx=5, pady=1)
        self.source_dir_button.grid(row=0, column=2, padx=5, pady=1)

        # Row 1: Patient Info Destination
        self.organize_by_patient_var = tk.BooleanVar(value=True)
        self.dest_dir_check = tk.Checkbutton(self.folder_frame, variable=self.organize_by_patient_var)
        self.dest_dir_label = tk.Label(self.folder_frame, text="Patient Info Destination:")
        self.dest_dir_entry = tk.Entry(self.folder_frame, width=50)
        self.dest_dir_button = tk.Button(self.folder_frame, text="Browse...", command=self.browse_dest_dir)
        self.dest_dir_label.grid(row=1, column=0, padx=5, pady=1, sticky="w")
        self.dest_dir_entry.grid(row=1, column=1, padx=5, pady=1)
        self.dest_dir_button.grid(row=1, column=2, padx=5, pady=1)
        self.dest_dir_check.grid(row=1, column=3, sticky="w")

        # Row 2: Date-based Destination
        self.organize_by_date_var = tk.BooleanVar(value=True)
        self.date_dest_dir_check = tk.Checkbutton(self.folder_frame, variable=self.organize_by_date_var)
        self.date_dest_dir_label = tk.Label(self.folder_frame, text="Date-based Destination:")
        self.date_dest_dir_entry = tk.Entry(self.folder_frame, width=50)
        self.date_dest_dir_button = tk.Button(self.folder_frame, text="Browse...", command=self.browse_date_dest_dir)
        self.date_dest_dir_label.grid(row=2, column=0, padx=5, pady=1, sticky="w")
        self.date_dest_dir_entry.grid(row=2, column=1, padx=5, pady=1)
        self.date_dest_dir_button.grid(row=2, column=2, padx=5, pady=1)
        self.date_dest_dir_check.grid(row=2, column=3, sticky="w")

        # Row 3: Organizer Name
        self.organizer_name_label = tk.Label(self.folder_frame, text="Organizer Name:")
        self.organizer_name_entry = tk.Entry(self.folder_frame, width=50)
        self.organizer_name_label.grid(row=3, column=0, padx=5, pady=1, sticky="w")
        self.organizer_name_entry.grid(row=3, column=1, padx=5, pady=1)

        self.control_frame = tk.Frame(root, padx=5, pady=2)
        self.control_frame.pack(padx=5, pady=2)

        self.button_frame = tk.Frame(self.control_frame)
        self.button_frame.pack(pady=1)

        self.start_button = tk.Button(self.button_frame, text="Start Organizing", command=self.start_organizing, state=tk.DISABLED)
        self.refresh_db_button = tk.Button(self.button_frame, text="Refresh DB", command=self.refresh_patient_db)
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_organizing, state=tk.DISABLED)
        self.settings_button = tk.Button(self.button_frame, text="Settings", command=self.open_settings_dialog)
        self.exit_button = tk.Button(self.button_frame, text="Exit", command=self.on_exit)
        
        self.start_button.pack(side=tk.LEFT, padx=5, pady=1)
        self.refresh_db_button.pack(side=tk.LEFT, padx=5, pady=1)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=1)
        self.settings_button.pack(side=tk.LEFT, padx=5, pady=1)
        self.exit_button.pack(side=tk.LEFT, padx=5, pady=1)

        self.review_not_info_var = tk.BooleanVar()
        self.review_not_info_checkbox = tk.Checkbutton(self.control_frame, text="모든 사진 확인하고 분류하기", variable=self.review_not_info_var)
        self.review_not_info_checkbox.pack(pady=1)

        self.organization_info_frame = tk.Frame(root)
        self.organization_info_frame.pack(fill="x", padx=5, pady=2)

        self.start_point_frame = tk.LabelFrame(self.organization_info_frame, text="Organize from:", padx=5, pady=2)
        self.start_point_frame.pack(side=tk.LEFT, padx=(0, 5), anchor="n")

        self.start_mode_var = tk.StringVar(value="auto")
        self.auto_mode_radio = tk.Radiobutton(self.start_point_frame, text="Last completed run", variable=self.start_mode_var, value="auto", command=self._toggle_manual_date_entry)
        self.auto_mode_radio.pack(anchor="w", pady=1)
        self.last_organized_label = tk.Label(self.start_point_frame, text="Last organized: Never")
        self.last_organized_label.pack(anchor="w", padx=20, pady=1)
        self.manual_mode_radio = tk.Radiobutton(self.start_point_frame, text="Specify date (YYYY-MM-DD)", variable=self.start_mode_var, value="manual", command=self._toggle_manual_date_entry)
        self.manual_mode_radio.pack(anchor="w", pady=1)
        self.manual_date_entry = tk.Entry(self.start_point_frame, width=40, state=tk.DISABLED)
        self.manual_date_entry.pack(anchor="w", padx=20, pady=1)

        self.model_info_frame = tk.LabelFrame(self.organization_info_frame, text="Program Info", padx=5, pady=2)
        self.model_info_frame.pack(side=tk.LEFT, fill="both", expand=True)
        info_text = ("환자정보/임상사진 분류 정확도: 99.61%\nConfidence 95% 미만인 임상 사진은 자동 체크\n제작일: 2025-07-24\n제작자: 경북대 치주과 최수영")
        tk.Label(self.model_info_frame, text=info_text, justify=tk.LEFT).pack(anchor="w", padx=5, pady=5)

        self.log_frame = tk.Frame(root, padx=5, pady=2)
        self.log_frame.pack(padx=5, pady=2, fill="both", expand=True)
        self.log_label = tk.Label(self.log_frame, text="Log:")
        self.log_label.pack(anchor="w")
        self.log_text_frame = tk.Frame(self.log_frame)
        self.log_text_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(self.log_text_frame, height=15, width=80)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scrollbar = tk.Scrollbar(self.log_text_frame, command=self.log_text.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.log_scrollbar.set)

        self.root.after(100, self.deferred_initialization)
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

    def deferred_initialization(self):
        self._log("Deferred initialization started.")
        self.load_all_settings()
        model_load_thread = threading.Thread(target=self._load_model_async)
        model_load_thread.daemon = True
        model_load_thread.start()
        self.refresh_patient_db()
        self._log("Initialization complete.")

    def _log(self, message):
        def append_message():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        if self.root and self.log_text.winfo_exists():
            self.root.after(0, append_message)

    def get_default_settings(self):
        return {
            'source_dir': os.path.join(application_path, "원본 폴더"),
            'dest_dir': os.path.join(application_path, "환자별 분류 폴더"),
            'date_dest_dir': os.path.join(application_path, "날짜별 분류 폴더"),
            'organizer_name': "",
            'last_organizing_timestamp': 0.0,
            'last_sync_timestamp': 0.0,
            'folder_structure': [{'format': '%Y년도'}, {'format': '%m월'}, {'format': '%Y.%m.%d ({weekday})'}, {'format': '{organizer_name}'}, {'format': '{patient_folder_name}'}],
            'file_naming_format': "%Y-%m-%d_{original_name}",
            'organize_by_patient': True,
            'organize_by_date': True
        }

    def load_all_settings(self):
        try:
            if os.path.exists(PATHS_FILE):
                with open(PATHS_FILE, 'r', encoding='utf-8') as f:
                    self.all_settings = json.load(f)
            else:
                self._log("paths.json not found. Creating default preset.")
                default_settings = self.get_default_settings()
                self.all_settings = {
                    "current_profile": "경북대 치주과",
                    "profiles": {
                        "경북대 치주과": default_settings
                    }
                }
                for key in ['source_dir', 'dest_dir', 'date_dest_dir']:
                    os.makedirs(default_settings[key], exist_ok=True)
                self.write_settings_to_disk()

            self.update_profile_combobox()
            self.switch_to_profile(self.all_settings.get("current_profile"))

        except Exception as e:
            self._log(f"Error loading settings: {e}")
            messagebox.showerror("Error", f"A critical error occurred while loading settings: {e}")

    def write_settings_to_disk(self):
        try:
            with open(PATHS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.all_settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self._log(f"Error saving settings to disk: {e}")

    def switch_to_profile(self, profile_name):
        if not profile_name or profile_name not in self.all_settings.get("profiles", {}):
            self._log(f"Attempted to switch to a non-existent profile: {profile_name}")
            available_profiles = list(self.all_settings.get("profiles", {}).keys())
            if not available_profiles:
                messagebox.showerror("Error", "No profiles available to load.")
                return
            profile_name = available_profiles[0]
            self._log(f"Falling back to first available profile: {profile_name}")

        self._log(f"Switching to profile: {profile_name}")
        profile_data = self.all_settings["profiles"][profile_name]
        default_profile = self.get_default_settings()

        self.source_dir_entry.delete(0, tk.END)
        self.source_dir_entry.insert(0, profile_data.get('source_dir', default_profile['source_dir']))
        self.dest_dir_entry.delete(0, tk.END)
        self.dest_dir_entry.insert(0, profile_data.get('dest_dir', default_profile['dest_dir']))
        self.date_dest_dir_entry.delete(0, tk.END)
        self.date_dest_dir_entry.insert(0, profile_data.get('date_dest_dir', default_profile['date_dest_dir']))
        self.organizer_name_entry.delete(0, tk.END)
        self.organizer_name_entry.insert(0, profile_data.get('organizer_name', default_profile['organizer_name']))
        
        self.organize_by_patient_var.set(profile_data.get('organize_by_patient', True))
        self.organize_by_date_var.set(profile_data.get('organize_by_date', True))

        self.last_organizing_timestamp = profile_data.get('last_organizing_timestamp', 0.0)
        self.last_sync_timestamp = profile_data.get('last_sync_timestamp', 0.0)
        self.folder_structure_config = profile_data.get('folder_structure', default_profile['folder_structure'])
        self.file_naming_format = profile_data.get('file_naming_format', default_profile['file_naming_format'])

        if self.last_organizing_timestamp > 0:
            self.last_organized_label.config(text=f"Last organized: {datetime.fromtimestamp(self.last_organizing_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            self.last_organized_label.config(text="Last organized: Never")
        
        self.all_settings["current_profile"] = profile_name
        self.profile_combobox.set(profile_name)

    def update_profile_combobox(self):
        profiles = list(self.all_settings.get("profiles", {}).keys())
        self.profile_combobox['values'] = profiles
        current = self.all_settings.get("current_profile")
        if current in profiles:
            self.profile_combobox.set(current)
        elif profiles:
            self.profile_combobox.set(profiles[0])

    def on_profile_select(self, event=None):
        selected_profile = self.profile_combobox.get()
        self.switch_to_profile(selected_profile)

    def save_current_profile(self):
        current_profile_name = self.profile_combobox.get()
        if not current_profile_name:
            messagebox.showwarning("Warning", "No profile selected to save.")
            return

        self._log(f"Saving settings to profile: {current_profile_name}")
        
        current_settings = {
            'source_dir': self.source_dir_entry.get(),
            'dest_dir': self.dest_dir_entry.get(),
            'date_dest_dir': self.date_dest_dir_entry.get(),
            'organizer_name': self.organizer_name_entry.get(),
            'last_organizing_timestamp': self.last_organizing_timestamp,
            'last_sync_timestamp': self.last_sync_timestamp,
            'folder_structure': self.folder_structure_config,
            'file_naming_format': self.file_naming_format,
            'organize_by_patient': self.organize_by_patient_var.get(),
            'organize_by_date': self.organize_by_date_var.get()
        }

        self.all_settings["profiles"][current_profile_name] = current_settings
        self.write_settings_to_disk()
        self._log(f"Profile '{current_profile_name}' saved successfully.")

    def new_profile(self):
        new_name = simpledialog.askstring("New Preset", "Enter a name for the new preset:", parent=self.root)
        if not new_name or not new_name.strip():
            return
        new_name = new_name.strip()

        if new_name in self.all_settings.get("profiles", {}):
            messagebox.showwarning("Duplicate", f"A preset named '{new_name}' already exists.")
            return

        self._log(f"Creating new profile: {new_name}")
        current_profile_settings = copy.deepcopy(self.all_settings["profiles"][self.profile_combobox.get()])
        self.all_settings["profiles"][new_name] = current_profile_settings
        self.all_settings["current_profile"] = new_name
        
        self.update_profile_combobox()
        self.switch_to_profile(new_name)
        self.write_settings_to_disk()

    def delete_profile(self):
        current_profile_name = self.profile_combobox.get()
        if not current_profile_name:
            messagebox.showwarning("Warning", "No profile selected to delete.")
            return

        if len(self.all_settings["profiles"]) <= 1:
            messagebox.showwarning("Warning", "Cannot delete the last preset.")
            return

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the preset '{current_profile_name}'?"):
            self._log(f"Deleting profile: {current_profile_name}")
            del self.all_settings["profiles"][current_profile_name]
            self.all_settings["current_profile"] = list(self.all_settings["profiles"].keys())[0]
            self.update_profile_combobox()
            self.switch_to_profile(self.all_settings["current_profile"])
            self.write_settings_to_disk()

    def open_settings_dialog(self):
        dialog = SettingsDialog(self.root, self.folder_structure_config, self.file_naming_format, self.on_settings_save)

    def on_settings_save(self, new_structure, new_file_format):
        self.folder_structure_config = new_structure
        self.file_naming_format = new_file_format
        self.save_current_profile()
        self._log("Folder and file settings saved to the current preset.")

    def _generate_dynamic_path(self, base_dir, dt_obj, context):
        path_parts = [base_dir]
        format_context = {
            'organizer_name': context.get('organizer_name', ''),
            'patient_folder_name': context.get('patient_folder_name', ''),
            'weekday': get_korean_weekday(dt_obj.weekday())
        }
        for level in self.folder_structure_config:
            part = dt_obj.strftime(level['format'].format(**format_context))
            path_parts.append(part)
        return os.path.join(*path_parts)

    def _generate_new_filename(self, dt_obj, original_path, context, counter):
        base_name, ext = os.path.splitext(os.path.basename(original_path))
        format_context = {
            'original_name': base_name,
            'weekday': get_korean_weekday(dt_obj.weekday()),
            'organizer_name': context.get('organizer_name', ''),
            'patient_folder_name': context.get('patient_folder_name', ''),
            'counter': str(counter).zfill(3)
        }
        new_base_name = dt_obj.strftime(self.file_naming_format.format(**format_context))
        return new_base_name + ext

    def on_exit(self):
        self._log("Saving current settings on exit...")
        self.save_current_profile()
        self.root.quit()

    def refresh_patient_db(self):
        dest_dir = self.dest_dir_entry.get()
        if not os.path.isdir(dest_dir):
            self.root.after(10, lambda: messagebox.showerror("Error", "Patient Info Destination directory does not exist or is not set."))
            return
        self._log("Refreshing patient database in the background...")
        db_load_thread = threading.Thread(target=self._load_patient_db_async, args=(dest_dir,))
        db_load_thread.daemon = True
        db_load_thread.start()

    def _load_patient_db_async(self, dest_dir):
        try:
            patient_list = self._scan_patient_info_from_disk(dest_dir)
            with self.patient_db_lock:
                self.patient_info_cache = patient_list
            self._log(f"Patient database refreshed. Found {len(patient_list)} unique patients.")
        except Exception as e:
            self._log(f"Error refreshing patient database: {e}")

    def _load_model_async(self):
        try:
            self._log("Loading deep learning model...")
            model = build_model(num_classes=2)
            model_path = resource_path('photo_classifier_model.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                self.model = model
                self._log("Model loaded successfully.")
                self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            else:
                self.model = None
                self._log(f"Error: Model file '{model_path}' not found.")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Model file '{model_path}' not found. Please ensure it's in the same directory."))
        except Exception as e:
            self.model = None
            error_message = f"An error occurred while loading the model: {e}"
            self._log(f"Error loading model: {e}")
            self.root.after(0, lambda: messagebox.showerror("Model Error", error_message))

    def _get_existing_patient_info(self, use_cache=True):
        if use_cache and self.patient_info_cache is not None:
            with self.patient_db_lock:
                return self.patient_info_cache.copy()
        dest_dir = self.dest_dir_entry.get()
        if not os.path.isdir(dest_dir):
            return []
        return self._scan_patient_info_from_disk(dest_dir)

    def _add_patient_to_cache(self, name, patient_id, folder_name):
        if self.patient_info_cache is None: return
        with self.patient_db_lock:
            exists = any(p['name'].lower() == name.lower() and p['id'] == patient_id for p in self.patient_info_cache)
            if not exists:
                self.patient_info_cache.append({"name": name, "id": patient_id, "folder_name": folder_name})
                self._log(f"Added new patient to cache: {name} ({patient_id})")

    def _scan_patient_info_from_disk(self, dest_dir):
        """Scans the destination directory on disk to build the patient list."""
        unique_patients = {}
        if not os.path.isdir(dest_dir): return []
        for folder_name in os.listdir(dest_dir):
            parts = folder_name.split()
            if len(parts) < 2:
                continue

            name = ""
            patient_id = ""
            id_index = -1

            # Find the first numeric part (the ID)
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) >= 4:
                    patient_id = part
                    id_index = i
                    break
            
            if id_index == -1: continue # No ID found

            # If ID is the first part, name must be the second
            if id_index == 0:
                name = parts[1]
            # If ID is not the first part, name must be right before it
            else:
                name = parts[id_index - 1]

            if not name: continue

            # Use a consistent key to handle variations in folder naming and capitalization.
            patient_key = (name.lower(), patient_id)
            if patient_key not in unique_patients:
                unique_patients[patient_key] = {"name": name, "id": patient_id, "folder_name": folder_name}
        return list(unique_patients.values())

    def browse_source_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.source_dir_entry.delete(0, tk.END)
            self.source_dir_entry.insert(0, directory)

    def browse_dest_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dest_dir_entry.delete(0, tk.END)
            self.dest_dir_entry.insert(0, directory)

    def browse_date_dest_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.date_dest_dir_entry.delete(0, tk.END)
            self.date_dest_dir_entry.insert(0, directory)

    def _toggle_manual_date_entry(self):
        self.manual_date_entry.config(state=tk.NORMAL if self.start_mode_var.get() == "manual" else tk.DISABLED)

    def start_organizing(self):
        try:
            source_dir = self.source_dir_entry.get()
            dest_dir = self.dest_dir_entry.get()
            date_dest_dir = self.date_dest_dir_entry.get()
            organizer_name = self.organizer_name_entry.get().strip()

            if not all([organizer_name, os.path.isdir(source_dir), os.path.isdir(dest_dir), os.path.isdir(date_dest_dir)]):
                messagebox.showerror("Input Error", "Please ensure all paths and the Organizer Name are set correctly.")
                return

            if not self.organize_by_patient_var.get() and not self.organize_by_date_var.get():
                messagebox.showerror("Input Error", "Please select at least one destination to organize into.")
                return

            start_timestamp = 0.0
            if self.start_mode_var.get() == "auto":
                start_timestamp = self.last_organizing_timestamp
            else:
                manual_date_str = self.manual_date_entry.get().strip()
                if not manual_date_str:
                    messagebox.showerror("Input Error", "Please enter a date for manual start.")
                    return
                try:
                    start_datetime = datetime.strptime(manual_date_str, '%Y-%m-%d')
                    start_timestamp = start_datetime.timestamp()
                except ValueError:
                    messagebox.showerror("Input Error", "Invalid date format. Please use YYYY-MM-DD.")
                    return
            
            self.is_organizing = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self._log(f"Starting organization...")
            
            organizing_thread = threading.Thread(target=self._start_organizing_thread, args=(source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp))
            organizing_thread.daemon = True
            organizing_thread.start()

        except Exception as e:
            self._log(f"Error in start_organizing: {e}")
            messagebox.showerror("Critical Error", f"An unexpected error occurred: {e}")
            self.is_organizing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def _start_organizing_thread(self, source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp):
        try:
            self.classify_and_organize(source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp)
            current_profile_name = self.profile_combobox.get()
            if current_profile_name:
                self.all_settings["profiles"][current_profile_name]['last_organizing_timestamp'] = datetime.now().timestamp()
                self.write_settings_to_disk()
        except Exception as e:
            self._log(f"Error in organizing thread: {e}")
            traceback.print_exc()
        finally:
            self.is_organizing = False
            self.root.after(0, self._finish_organization)

    def _finish_organization(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self._log("Organization process finished or stopped.")

    def stop_organizing(self):
        self.is_organizing = False
        self._log("Stopping organization process...")

    def _collect_image_files(self, source_dir, start_timestamp):
        self._log(f"Scanning source directory: {source_dir}")
        all_image_files = []
        for filename in sorted(os.listdir(source_dir)):
            file_path = os.path.join(source_dir, filename)
            if not os.path.isfile(file_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            if os.path.getmtime(file_path) > start_timestamp:
                all_image_files.append(file_path)
        self._log(f"Found {len(all_image_files)} images to classify.")
        return all_image_files

    def classify_and_organize(self, source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp):
        start_time = time.time()
        all_image_files = self._collect_image_files(source_dir, start_timestamp)
        if not self.is_organizing: self._log("Organization stopped."); return
        if self.model is None: self._log("Model not loaded. Aborting."); return

        classified_results = {}
        self.model.eval()
        for image_path in all_image_files:
            if not self.is_organizing: self._log("Organization stopped."); return
            try:
                image = PIL.Image.open(image_path).convert('RGB')
                input_tensor = preprocess(image)
                input_batch = input_tensor.unsqueeze(0)
                with torch.no_grad():
                    output = self.model(input_batch.to(DEVICE))
                    confidence = torch.sigmoid(output).item()
                    predicted_class = "not_info" if confidence > 0.5 else "info"
                classified_results[image_path] = {"class": predicted_class, "confidence": confidence}
            except Exception as e:
                self._log(f"Error classifying {os.path.basename(image_path)}: {e}")
                classified_results[image_path] = {"class": "not_info", "confidence": 0.0}

        info_images = [p for p, data in classified_results.items() if data["class"] == "info"]
        not_info_images = [p for p, data in classified_results.items() if data["class"] == "not_info"]

        if self.review_not_info_var.get():
            images_to_change_to_info = []
            for image_path in not_info_images:
                if not self.is_organizing: self._log("Organization stopped."); return
                action = self._prompt_for_image_review(image_path, "not_info", classified_results.get(image_path, {}).get("confidence", 0.0))
                if action == "change_to_info":
                    images_to_change_to_info.append(image_path)
            if images_to_change_to_info:
                not_info_images = [p for p in not_info_images if p not in images_to_change_to_info]
                info_images.extend(images_to_change_to_info)

        manual_processed_info = {}
        for image_path in info_images:
            if not self.is_organizing: self._log("Organization stopped."); return
            patient_name, patient_id, classified_as_clinical_by_user = self.get_patient_info(image_path)
            if classified_as_clinical_by_user:
                not_info_images.append(image_path)
            elif patient_name and patient_id:
                manual_processed_info[image_path] = (patient_name, patient_id)

        patient_info_count = len(manual_processed_info)
        clinical_count = len(not_info_images)
        current_patient_context_name = None
        all_images_sorted = sorted(all_image_files, key=lambda p: os.path.getmtime(p))
        
        file_counter = 0
        for image_path in all_images_sorted:
            if not self.is_organizing: self._log("Organization stopped."); return
            file_counter += 1
            try:
                image = PIL.Image.open(image_path)
                exif_data = image._getexif()
                creation_datetime = datetime.strptime(exif_data[36867], '%Y:%m:%d %H:%M:%S') if exif_data and 36867 in exif_data else datetime.fromtimestamp(os.path.getmtime(image_path))
            except Exception:
                creation_datetime = datetime.fromtimestamp(os.path.getmtime(image_path))

            path_context = {'organizer_name': organizer_name}
            if image_path in manual_processed_info:
                patient_name, patient_id = manual_processed_info[image_path]
                folder_name_patient_id = f"{patient_name} {patient_id}"
                current_patient_context_name = folder_name_patient_id
                path_context['patient_folder_name'] = folder_name_patient_id
                new_filename = self._generate_new_filename(creation_datetime, image_path, path_context, file_counter)
                
                if self.organize_by_patient_var.get():
                    patient_folder_dest = os.path.join(dest_dir, folder_name_patient_id)
                    self._copy_image_to_single_destination(image_path, new_filename, patient_folder_dest)
                
                if self.organize_by_date_var.get():
                    target_patient_folder_in_date = self._generate_dynamic_path(date_dest_dir, creation_datetime, path_context)
                    self._copy_image_to_single_destination(image_path, new_filename, target_patient_folder_in_date)

            elif image_path in not_info_images:
                new_filename = self._generate_new_filename(creation_datetime, image_path, path_context, file_counter)
                if current_patient_context_name:
                    path_context['patient_folder_name'] = current_patient_context_name
                    if self.organize_by_patient_var.get():
                        patient_folder_dest = os.path.join(dest_dir, current_patient_context_name)
                        self._copy_image_to_single_destination(image_path, new_filename, patient_folder_dest)
                    if self.organize_by_date_var.get():
                        target_patient_folder_in_date = self._generate_dynamic_path(date_dest_dir, creation_datetime, path_context)
                        self._copy_image_to_single_destination(image_path, new_filename, target_patient_folder_in_date)
                else:
                    if self.organize_by_patient_var.get():
                        unclassified_folder_dest = os.path.join(dest_dir, "unclassified_clinical")
                        self._copy_image_to_single_destination(image_path, new_filename, unclassified_folder_dest)
                    if self.organize_by_date_var.get():
                        unclassified_folder_date_dest = os.path.join(date_dest_dir, "unclassified_clinical")
                        self._copy_image_to_single_destination(image_path, new_filename, unclassified_folder_date_dest)

        elapsed_time = time.time() - start_time
        summary_message = (f"\n--- Organization Complete ---\n"
                           f"Total elapsed time: {elapsed_time:.2f} seconds\n"
                           f"Patient info files processed: {patient_info_count}\n"
                           f"Clinical files processed: {clinical_count}\n"
                           f"-----------------------------\n")
        self._log(summary_message)

    def get_patient_info(self, image_path):
        if threading.current_thread() != threading.main_thread():
            q = queue.Queue()
            self.root.after(0, lambda: q.put(self._show_patient_info_dialog(image_path)))
            return q.get()
        return self._show_patient_info_dialog(image_path)

    def _show_patient_info_dialog(self, image_path):
        dialog = tk.Toplevel(self.root)
        dialog.title("Enter Patient Information")
        dialog.geometry(f"500x620+{self.root.winfo_x()+150}+{self.root.winfo_y()+150}")
        dialog.transient(self.root)
        dialog.grab_set()

        style = ttk.Style(dialog)
        style.configure("Blue.TCombobox", foreground="blue")

        try:
            img_pil = PIL.Image.open(image_path).convert('RGB')
            img_pil.thumbnail((400, 400))
            photo = PIL.ImageTk.PhotoImage(img_pil)
            img_label = tk.Label(dialog, image=photo)
            img_label.image = photo
            img_label.pack(pady=10)
        except Exception as e:
            tk.Label(dialog, text=f"Could not load image: {e}").pack(pady=10)

        tk.Label(dialog, text="Patient Name:").pack()
        name_entry = tk.Entry(dialog, width=40)
        name_entry.pack()
        name_entry.focus_set()

        tk.Label(dialog, text="Patient ID:").pack()
        id_entry = ttk.Combobox(dialog, width=38, style="TCombobox")
        id_entry.pack()

        result = {"name": None, "id": None, "classified_as_clinical": False}
        id_selection_confirmed = False

        def _confirm_id_selection(event=None):
            nonlocal id_selection_confirmed
            id_selection_confirmed = True
            id_entry.configure(style="TCombobox")

        def _clear_name_entry(_event=None):
            name_entry.delete(0, tk.END)
            name_entry.unbind("<Button-1>")

        def _reset_name_color(_event=None):
            name_entry.config(fg="black")
            name_entry.unbind("<KeyPress>")

        def _autocomplete_id(_event=None):
            nonlocal id_selection_confirmed
            id_selection_confirmed = False
            id_entry.configure(style="TCombobox")
            name_typed = name_entry.get().strip()
            id_entry.set('')
            id_entry.config(values=[])
            if not name_typed: return
            matching_patients = [p for p in self._get_existing_patient_info() if p["name"].lower() == name_typed.lower()]
            if matching_patients:
                ids = [p["id"] for p in matching_patients]
                id_entry.config(values=ids)
                id_entry.set(ids[0])
                id_entry.configure(style="Blue.TCombobox")
                if len(ids) == 1:
                    id_selection_confirmed = True
            else:
                id_selection_confirmed = True

        def _autocomplete_name(_event=None):
            id_typed = id_entry.get().strip()
            if not id_typed or name_entry.get(): return
            for patient in self._get_existing_patient_info():
                if patient["id"].lower() == id_typed.lower():
                    name_entry.delete(0, tk.END)
                    name_entry.insert(0, patient["name"])
                    name_entry.config(fg="blue")
                    name_entry.bind("<KeyPress>", _reset_name_color)
                    break

        def on_ok():
            nonlocal id_selection_confirmed
            name = name_entry.get().strip()
            patient_id = id_entry.get().strip()
            values_in_list = dialog.tk.splitlist(id_entry.cget('values'))
            if len(values_in_list) > 1 and not id_selection_confirmed:
                messagebox.showwarning("Confirm Selection", "동일한 이름의 환자가 여러 명 있습니다. ID를 확인하거나 목록에서 선택한 후 다시 시도해주세요.", parent=dialog)
                return
            if name.isdigit() and not patient_id.isdigit():
                name, patient_id = patient_id, name
            if not name or not patient_id:
                messagebox.showwarning("Input Error", "Patient Name and ID cannot be empty.", parent=dialog)
                return
            result["name"], result["id"] = name, patient_id
            dialog.destroy()

        def on_classify_as_clinical():
            result["classified_as_clinical"] = True
            dialog.destroy()

        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Classify as Clinical", command=on_classify_as_clinical).pack(side=tk.RIGHT, padx=10)

        name_entry.bind("<Button-1>", _clear_name_entry)
        name_entry.bind("<FocusOut>", _autocomplete_id)
        name_entry.bind("<Return>", lambda e: id_entry.focus_set())
        id_entry.bind("<FocusOut>", _autocomplete_name)
        id_entry.bind("<Return>", lambda e: on_ok())
        id_entry.bind("<<ComboboxSelected>>", _confirm_id_selection)
        id_entry.bind("<KeyPress>", _confirm_id_selection)

        self.root.wait_window(dialog)

        if not result["classified_as_clinical"] and result["name"] and result["id"]:
            existing_patients = self._get_existing_patient_info()
            exact_match_found = any(p["name"].lower() == result["name"].lower() and p["id"].lower() == result["id"].lower() for p in existing_patients)
            if not exact_match_found:
                self._add_patient_to_cache(result["name"], result["id"], f"{result['name']} {result['id']}")
                similar_suggestions = self._find_similar_patients(result["name"], result["id"], existing_patients)
                if similar_suggestions:
                    chosen_name, chosen_id = self._prompt_for_suggestion(similar_suggestions, result["name"], result["id"])
                    result["name"], result["id"] = chosen_name, chosen_id
        return result["name"], result["id"], result["classified_as_clinical"]

    def _prompt_for_image_review(self, image_path, predicted_class, confidence):
        if threading.current_thread() != threading.main_thread():
            q = queue.Queue()
            self.root.after(0, lambda: q.put(self._show_image_review_dialog(image_path, predicted_class, confidence)))
            return q.get()
        else:
            return self._show_image_review_dialog(image_path, predicted_class, confidence)

    def _show_image_review_dialog(self, image_path, predicted_class, confidence):
        dialog = tk.Toplevel(self.root)
        dialog.title("Image Review")
        dialog.geometry(f"450x550+{self.root.winfo_x()+100}+{self.root.winfo_y()+100}")
        dialog.transient(self.root)
        dialog.grab_set()
        try:
            img = PIL.Image.open(image_path)
            img.thumbnail((400, 400))
            photo = PIL.ImageTk.PhotoImage(img)
            img_label = tk.Label(dialog, image=photo)
            img_label.image = photo
            img_label.pack(padx=10, pady=10)
        except Exception as e:
            tk.Label(dialog, text=f"Could not load image: {e}").pack(padx=10, pady=10)
        info_text = f"File: {os.path.basename(image_path)}\nPredicted: {predicted_class} (Confidence: {confidence:.2f})"
        tk.Label(dialog, text=info_text).pack(padx=10, pady=5)
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        result = {"action": "keep"}
        def on_change_to_info():
            result["action"] = "change_to_info"
            dialog.destroy()
        def on_keep_as_not_info():
            result["action"] = "keep"
            dialog.destroy()
        if predicted_class == "not_info":
            tk.Button(button_frame, text="Change to 'Patient Info'", command=on_change_to_info).pack(side=tk.LEFT, padx=10)
            tk.Button(button_frame, text="Keep as 'Clinical'", command=on_keep_as_not_info).pack(side=tk.RIGHT, padx=10)
        else:
            tk.Button(button_frame, text="This is a 'Patient Info' file", command=lambda: dialog.destroy()).pack(padx=10)
        self.root.wait_window(dialog)
        return result["action"]

    def _find_similar_patients(self, input_name, input_id, existing_patients):
        similar_suggestions = []
        for patient in existing_patients:
            name_similarity = difflib.SequenceMatcher(None, input_name.lower(), patient["name"].lower()).ratio()
            id_similarity = difflib.SequenceMatcher(None, input_id.lower(), patient["id"].lower()).ratio()
            if (id_similarity == 1.0 and name_similarity < 1.0) or \
               (name_similarity > 0.8 and id_similarity > 0.9) or \
               (name_similarity > 0.9 and id_similarity > 0.8):
                similar_suggestions.append(patient)
        return similar_suggestions

    def _prompt_for_suggestion(self, suggestions, original_name, original_id):
        if threading.current_thread() != threading.main_thread():
            q = queue.Queue()
            self.root.after(0, lambda: q.put(self._show_suggestion_dialog(suggestions, original_name, original_id)))
            return q.get()
        else:
            return self._show_suggestion_dialog(suggestions, original_name, original_id)

    def _show_suggestion_dialog(self, suggestions, original_name, original_id):
        suggestion_window = tk.Toplevel(self.root)
        suggestion_window.title("Did you mean...?)")
        suggestion_window.geometry(f"400x300+{self.root.winfo_x()+700}+{self.root.winfo_y()+150}")
        suggestion_window.transient(self.root)
        suggestion_window.grab_set()
        tk.Label(suggestion_window, text="Did you mean one of these existing patients?").pack(pady=10)
        selected_option = tk.StringVar(value="original")
        for i, patient in enumerate(suggestions):
            text = f"{patient['name']} {patient['id']}"
            tk.Radiobutton(suggestion_window, text=text, variable=selected_option, value=f"suggestion_{i}").pack(anchor="w", padx=20)
        tk.Radiobutton(suggestion_window, text=f"No, use my input: {original_name} {original_id}", variable=selected_option, value="original").pack(anchor="w", padx=20)
        result = {"name": original_name, "id": original_id}
        def on_select():
            choice = selected_option.get()
            if choice.startswith("suggestion_"):
                index = int(choice.split("_")[1])
                result["name"] = suggestions[index]["name"]
                result["id"] = suggestions[index]["id"]
            suggestion_window.destroy()
        tk.Button(suggestion_window, text="Select", command=on_select).pack(pady=10)
        self.root.wait_window(suggestion_window)
        return result["name"], result["id"]

    def _copy_image_to_single_destination(self, source_path, new_filename, dest_folder):
        try:
            os.makedirs(dest_folder, exist_ok=True)
            dest_path = os.path.join(dest_folder, new_filename)
            if not os.path.exists(dest_path):
                shutil.copy2(source_path, dest_path)
                self._log(f"Copied {os.path.basename(source_path)} to {dest_path}")
            else:
                self._log(f"Skipped existing file: {dest_path}")
        except Exception as e:
            self._log(f"Error copying {os.path.basename(source_path)} to {dest_folder}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoOrganizerApp(root)
    root.mainloop()
