import sys
import os
import traceback

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

    print("--- Application Start ---")

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


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- 딥러닝 모델 설정 ---
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cpu")

# Determine the base path for writable files like paths.json
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the base path is the directory of the executable
    application_path = os.path.dirname(sys.executable)
else:
    # If run as a script, the base path is the script's directory
    application_path = os.path.abspath(".")
PATHS_FILE = os.path.join(application_path, 'paths.json')

def build_model(num_classes):
    """MobileNetV2를 기반으로 한 전이 학습 모델을 구축합니다."""
    # weights=None으로 설정하여 미리 학습된 가중치를 로드하지 않음
    model = models.mobilenet_v2(weights=None, progress=False)

    # 번들링된 mobilenet_v2-7ebf99e0.pth 파일에서 가중치를 로드
    # 이 파일은 PyInstaller의 datas 섹션에 추가되어야 합니다.
    mobilenet_weights_path = resource_path('mobilenet_v2-7ebf99e0.pth')
    if os.path.exists(mobilenet_weights_path):
        state_dict = torch.load(mobilenet_weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        # 인터넷 다운로드 시도 대신 명확한 오류 메시지 출력
        raise FileNotFoundError(f"MobileNetV2 weights file not found at {mobilenet_weights_path}. Please ensure it's bundled correctly.")

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 1) # Output feature of 1 for binary classification with sigmoid
    )
    
    return model.to(DEVICE)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_korean_weekday(weekday):
    korean_weekdays = ['월', '화', '수', '목', '금', '토', '일']
    return korean_weekdays[weekday]

class PhotoOrganizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Organizer")
        self.root.geometry("800x500+100+100") # Set window size and position (width x height + x_offset + y_offset)
        self.root.resizable(False, False) # Prevent resizing

        self.is_organizing = False # Add this flag
        self.last_sync_timestamp = 0.0
        self.last_organizing_timestamp = 0.0
        self.model = None # Initialize model as None
        self.patient_info_cache = None # Cache for patient database
        self.patient_db_lock = threading.Lock() # Lock for cache access

        # Frame for folder selection
        self.folder_frame = tk.Frame(root, padx=5, pady=2)
        self.folder_frame.pack(padx=5, pady=2)

        self.source_dir_label = tk.Label(self.folder_frame, text="Source Directory:")
        self.source_dir_entry = tk.Entry(self.folder_frame, width=50)
        self.source_dir_button = tk.Button(self.folder_frame, text="Browse...", command=self.browse_source_dir)
        self.source_dir_label.grid(row=0, column=0, padx=5, pady=1, sticky="w")
        self.source_dir_entry.grid(row=0, column=1, padx=5, pady=1)
        self.source_dir_button.grid(row=0, column=2, padx=5, pady=1)

        self.dest_dir_label = tk.Label(self.folder_frame, text="Patient Info Destination:")
        self.dest_dir_entry = tk.Entry(self.folder_frame, width=50)
        self.dest_dir_button = tk.Button(self.folder_frame, text="Browse...", command=self.browse_dest_dir)
        self.dest_dir_label.grid(row=1, column=0, padx=5, pady=1, sticky="w")
        self.dest_dir_entry.grid(row=1, column=1, padx=5, pady=1)
        self.dest_dir_button.grid(row=1, column=2, padx=5, pady=1)

        self.date_dest_dir_label = tk.Label(self.folder_frame, text="Date-based Destination:")
        self.date_dest_dir_entry = tk.Entry(self.folder_frame, width=50)
        self.date_dest_dir_button = tk.Button(self.folder_frame, text="Browse...", command=self.browse_date_dest_dir)
        self.date_dest_dir_label.grid(row=2, column=0, padx=5, pady=1, sticky="w")
        self.date_dest_dir_entry.grid(row=2, column=1, padx=5, pady=1)
        self.date_dest_dir_button.grid(row=2, column=2, padx=5, pady=1)

        self.organizer_name_label = tk.Label(self.folder_frame, text="Organizer Name:")
        self.organizer_name_entry = tk.Entry(self.folder_frame, width=50)
        self.organizer_name_label.grid(row=3, column=0, padx=5, pady=1, sticky="w")
        self.organizer_name_entry.grid(row=3, column=1, padx=5, pady=1)

        # Frame for controls
        self.control_frame = tk.Frame(root, padx=5, pady=2)
        self.control_frame.pack(padx=5, pady=2)

        self.button_frame = tk.Frame(self.control_frame)
        self.button_frame.pack(pady=1)

        self.start_button = tk.Button(self.button_frame, text="Start Organizing", command=self.start_organizing, state=tk.DISABLED)
        self.refresh_db_button = tk.Button(self.button_frame, text="Refresh DB", command=self.refresh_patient_db)
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_organizing, state=tk.DISABLED)
        self.exit_button = tk.Button(self.button_frame, text="Exit", command=self.on_exit)
        
        self.start_button.pack(side=tk.LEFT, padx=5, pady=1)
        self.refresh_db_button.pack(side=tk.LEFT, padx=5, pady=1)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=1)
        self.exit_button.pack(side=tk.LEFT, padx=5, pady=1)

        self.review_not_info_var = tk.BooleanVar()
        self.review_not_info_checkbox = tk.Checkbutton(self.control_frame, text="모든 사진 확인하고 분류하기", variable=self.review_not_info_var)
        self.review_not_info_checkbox.pack(pady=1)

        # --- Container for Start Point and Model Info ---
        self.organization_info_frame = tk.Frame(root)
        self.organization_info_frame.pack(fill="x", padx=5, pady=2)

        # Frame for organizing start point selection
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

        # Frame for Deep Learning Model Info
        self.model_info_frame = tk.LabelFrame(self.organization_info_frame, text="Program Info", padx=5, pady=2)
        self.model_info_frame.pack(side=tk.LEFT, fill="both", expand=True)

        info_text = (
            "환자정보/임상사진 분류 정확도: 99.61%\n"
            "Confidence 95% 미만인 임상 사진은 자동 체크\n"
            "제작일: 2025-07-24\n"
            "제작자: 경북대 치주과 최수영"
        )
        tk.Label(self.model_info_frame, text=info_text, justify=tk.LEFT).pack(anchor="w", padx=5, pady=5)

        # Frame for logging
        self.log_frame = tk.Frame(root, padx=5, pady=2)
        self.log_frame.pack(padx=5, pady=2, fill="both", expand=True)

        self.log_label = tk.Label(self.log_frame, text="Log:")
        self.log_label.pack(anchor="w")

        # Create a frame for the Text widget and Scrollbar
        self.log_text_frame = tk.Frame(self.log_frame)
        self.log_text_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(self.log_text_frame, height=15, width=80)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.log_scrollbar = tk.Scrollbar(self.log_text_frame, command=self.log_text.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.log_scrollbar.set)

        # Load saved paths
        self.load_paths()

        # Save paths on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # Load model in background
        model_load_thread = threading.Thread(target=self._load_model_async)
        model_load_thread.daemon = True
        model_load_thread.start()

        # Initial patient DB load
        self.refresh_patient_db()

    def _log(self, message):
        """Logs a message to the text widget in a thread-safe way."""
        def append_message():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        
        # Ensure the GUI update is called from the main thread
        if self.root and self.log_text.winfo_exists():
            self.root.after(0, append_message)

    def refresh_patient_db(self):
        """(Re)loads the patient database from the destination directory in a background thread."""
        dest_dir = self.dest_dir_entry.get()
        if not os.path.isdir(dest_dir):
            messagebox.showerror("Error", "Patient Info Destination directory does not exist.")
            return

        self._log("Refreshing patient database in the background...")

        db_load_thread = threading.Thread(target=self._load_patient_db_async, args=(dest_dir,))
        db_load_thread.daemon = True
        db_load_thread.start()

    def _load_patient_db_async(self, dest_dir):
        """The actual logic to load the patient DB, run in a thread."""
        try:
            # This function performs the actual scan
            patient_list = self._scan_patient_info_from_disk(dest_dir)
            
            # Safely update the cache
            with self.patient_db_lock:
                self.patient_info_cache = patient_list
            
            self._log(f"Patient database refreshed. Found {len(patient_list)} unique patients.")
        except Exception as e:
            self._log(f"Error refreshing patient database: {e}")

    def _load_model_async(self):
        """Loads the deep learning model in a background thread."""
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
            self._log(f"Error loading model: {e}")
            self.root.after(0, lambda: messagebox.showerror("Model Error", f"An error occurred while loading the model: {e}"))

    def on_exit(self):
        self.save_paths()
        self.root.quit()

    def load_paths(self):
        try:
            if os.path.exists(PATHS_FILE):
                with open(PATHS_FILE, 'r') as f:
                    paths = json.load(f)
                    if 'source_dir' in paths:
                        self.source_dir_entry.delete(0, tk.END)
                        self.source_dir_entry.insert(0, paths['source_dir'])
                    if 'dest_dir' in paths:
                        self.dest_dir_entry.delete(0, tk.END)
                        self.dest_dir_entry.insert(0, paths['dest_dir'])
                    if 'date_dest_dir' in paths:
                        self.date_dest_dir_entry.delete(0, tk.END)
                        self.date_dest_dir_entry.insert(0, paths['date_dest_dir'])
                    if 'organizer_name' in paths:
                        self.organizer_name_entry.delete(0, tk.END)
                        self.organizer_name_entry.insert(0, paths['organizer_name'])
                    self.last_organizing_timestamp = paths.get('last_organizing_timestamp', 0.0) # Default to 0.0 if not found
                    self.last_sync_timestamp = paths.get('last_sync_timestamp', 0.0)
                    if self.last_organizing_timestamp > 0:
                        self.last_organized_label.config(text=f"Last organized: {datetime.fromtimestamp(self.last_organizing_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        self.last_organized_label.config(text="Last organized: Never")
        except Exception as e:
            self._log(f"Error loading paths: {e}")

    def save_paths(self):
        try:
            paths = {
                'source_dir': self.source_dir_entry.get(),
                'dest_dir': self.dest_dir_entry.get(),
                'date_dest_dir': self.date_dest_dir_entry.get(),
                'last_organizing_timestamp': self.last_organizing_timestamp,
                'last_sync_timestamp': self.last_sync_timestamp, # Save the sync timestamp
                'organizer_name': self.organizer_name_entry.get()
            }
            with open(PATHS_FILE, 'w') as f:
                json.dump(paths, f, indent=4)
        except Exception as e:
            self._log(f"Error saving paths: {e}")

    def _get_existing_patient_info(self, use_cache=True):
        """Gets patient info. If use_cache is True, returns the cached list. Otherwise, scans from disk."""
        if use_cache and self.patient_info_cache is not None:
            with self.patient_db_lock:
                return self.patient_info_cache.copy() # Return a copy to prevent modification
        
        # Fallback to scanning from disk if cache is not available or not desired
        dest_dir = self.dest_dir_entry.get()
        if not os.path.isdir(dest_dir):
            return []
        return self._scan_patient_info_from_disk(dest_dir)

    def _add_patient_to_cache(self, name, patient_id, folder_name):
        """Adds a new patient to the cache if they don't already exist."""
        if self.patient_info_cache is None:
            return # Cache not initialized yet

        with self.patient_db_lock:
            # Check for existence based on name and ID
            exists = any(p['name'].lower() == name.lower() and p['id'] == patient_id for p in self.patient_info_cache)
            if not exists:
                self.patient_info_cache.append({
                    "name": name,
                    "id": patient_id,
                    "folder_name": folder_name
                })
                self._log(f"Added new patient to cache: {name} ({patient_id})")

    def _scan_patient_info_from_disk(self, dest_dir):
        """Scans the destination directory on disk to build the patient list."""
        # Use a dictionary to store unique patients, keyed by a normalized (name, id) tuple.
        # This handles both 'Name ID' and 'ID Name' formats and merges them.
        unique_patients = {}
        if not os.path.isdir(dest_dir):
            return []

        for folder_name in os.listdir(dest_dir):
            # Try to extract a multi-digit number as the ID.
            id_match = re.search(r'(\b\d{4,}\b)', folder_name)
            if not id_match:
                continue

            patient_id = id_match.group(1)
            # The rest of the string is the name.
            name = folder_name.replace(patient_id, "").strip()
            # Clean up potential multiple spaces from the replacement.
            name = re.sub(r'\s+', ' ', name).strip()
            
            if not name:
                continue

            # Use a consistent key to handle variations in folder naming and capitalization.
            patient_key = (name.lower(), patient_id)
            if patient_key not in unique_patients:
                unique_patients[patient_key] = {
                    "name": name, 
                    "id": patient_id, 
                    "folder_name": folder_name
                }
        # Return the list of unique patient dictionaries.
        return list(unique_patients.values())

    def _find_similar_patients(self, input_name, input_id, existing_patients):
        similar_suggestions = []
        for patient in existing_patients:
            name_similarity = difflib.SequenceMatcher(None, input_name.lower(), patient["name"].lower()).ratio()
            id_similarity = difflib.SequenceMatcher(None, input_id.lower(), patient["id"].lower()).ratio()

            # Define thresholds for "similar"
            # Condition 1: Exact ID match, but name is not an exact match
            if (id_similarity == 1.0 and name_similarity < 1.0) or \
               (name_similarity > 0.8 and id_similarity > 0.9) or \
               (name_similarity > 0.9 and id_similarity > 0.8):
                similar_suggestions.append(patient)
        return similar_suggestions

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
        if self.start_mode_var.get() == "manual":
            self.manual_date_entry.config(state=tk.NORMAL)
        else:
            self.manual_date_entry.config(state=tk.DISABLED)

    def start_organizing(self):
        try: # Add try-except block for the entire function
            source_dir = self.source_dir_entry.get()
            dest_dir = self.dest_dir_entry.get()
            date_dest_dir = self.date_dest_dir_entry.get()
            organizer_name = self.organizer_name_entry.get().strip()

            if not organizer_name:
                messagebox.showerror("Input Error", "Organizer Name cannot be empty.")
                self._log("Error: Organizer Name is empty.")
                return

            self._log("start_organizing function called.")

            if not os.path.isdir(source_dir):
                messagebox.showerror("Error", "Source directory does not exist.")
                self._log("Error: Source directory does not exist.")
                return
            if not os.path.isdir(dest_dir):
                messagebox.showerror("Error", "Destination directory does not exist.")
                self._log("Error: Destination directory does not exist.")
                return
            if not os.path.isdir(date_dest_dir):
                messagebox.showerror("Error", "Date Destination directory does not exist.")
                self._log("Error: Date Destination directory does not exist.")
                return

            # Determine the start timestamp based on selected mode
            start_timestamp = 0.0
            if self.start_mode_var.get() == "auto":
                start_timestamp = self.last_organizing_timestamp
                self._log(f"Starting organization from last completed run: {datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
            else: # Manual mode
                manual_date_str = self.manual_date_entry.get().strip()
                if not manual_date_str:
                    messagebox.showerror("Input Error", "Please enter a date for manual start.")
                    return
                try:
                    start_datetime = datetime.strptime(manual_date_str, '%Y-%m-%d')
                    start_timestamp = start_datetime.timestamp()
                    self._log(f"Starting organization from manual date: {manual_date_str}")
                except ValueError:
                    messagebox.showerror("Input Error", "Invalid date format. Please use YYYY-MM-DD.")
                    return
            self.is_organizing = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self._log(f"Starting organization from {source_dir} to {dest_dir} and {date_dest_dir}")
            
            # Run the organizing process in a separate thread to keep GUI responsive
            self._log("Creating and starting organizing thread...")
            organizing_thread = threading.Thread(target=self._start_organizing_thread, args=(source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp))
            organizing_thread.daemon = True # Allow the main program to exit even if the thread is still running
            organizing_thread.start()
            self._log("Organizing thread started.")

        except Exception as e:
            self._log(f"Error in start_organizing function: {e}")
            messagebox.showerror("Critical Error", f"An unexpected error occurred: {e}")
            self.is_organizing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)


    def _start_organizing_thread(self, source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp):
        self._log("_start_organizing_thread started.")
        try:
            self.classify_and_organize(source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp)
            self.last_organizing_timestamp = datetime.now().timestamp() # Update timestamp after successful organization
            self.save_paths() # Save paths after organizing
        except Exception as e:
            self._log(f"Error in _start_organizing_thread: {e}")
            traceback.print_exc() # Print full traceback to log file
        finally:
            self.is_organizing = False
            # Schedule all final UI updates and the sync process on the main thread
            self.root.after(0, self._finish_organization)

    def _finish_organization(self):
        """Handles final UI updates and triggers synchronization. Runs on the main thread."""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self._log("Organization process finished or stopped.")
        
        # Now that the worker thread is done, run sync on the main thread.
        self.synchronize_folders()
            

    def stop_organizing(self):
        self.is_organizing = False
        self._log("Stopping organization process...")

    def _collect_image_files(self, source_dir, start_timestamp):
        """Scans the source directory and returns a list of image files to be processed."""
        self._log(f"Scanning source directory: {source_dir}")
        
        all_image_files = []
        for filename in sorted(os.listdir(source_dir)):
            file_path = os.path.join(source_dir, filename)
            if not os.path.isfile(file_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            if os.path.getmtime(file_path) > start_timestamp:
                all_image_files.append(file_path)
            else:
                self._log(f"Skipping {filename} (already organized or older than specified start time).")
        
        self._log(f"Found {len(all_image_files)} images to classify.")
        return all_image_files

    def classify_and_organize(self, source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp):
        start_time = time.time()
        self._log("classify_and_organize started.")

        all_image_files = self._collect_image_files(source_dir, start_timestamp)

        if not self.is_organizing: self._log("Organization stopped by user."); return

        classified_results = {}
        if self.model is None: self._log("Deep learning model not loaded. Aborting classification."); return

        self.model.eval()
        self._log("Starting image classification loop.")
        for image_path in all_image_files:
            if not self.is_organizing: self._log("Organization stopped by user."); return
            try:
                image = PIL.Image.open(image_path).convert('RGB')
                input_tensor = preprocess(image)
                input_batch = input_tensor.unsqueeze(0)

                with torch.no_grad():
                    output = self.model(input_batch.to(DEVICE))
                    confidence = torch.sigmoid(output).item()
                    predicted_class = "not_info" if confidence > 0.5 else "info"

                classified_results[image_path] = {"class": predicted_class, "confidence": confidence}
                self._log(f"Classified {os.path.basename(image_path)} as {predicted_class} (Confidence: {confidence:.2f})")

            except Exception as e:
                self._log(f"Error classifying image {os.path.basename(image_path)}: {e}")
                classified_results[image_path] = {"class": "not_info", "confidence": 0.0}

        self._log("Image classification loop finished. Starting organization based on results.")

        info_images = [p for p, data in classified_results.items() if data["class"] == "info"]
        not_info_images = [p for p, data in classified_results.items() if data["class"] == "not_info"]

        # --- Review 'not_info' images if the checkbox is ticked ---
        if self.review_not_info_var.get():
            images_to_change_to_info = []
            for image_path in not_info_images:
                if not self.is_organizing: self._log("Organization stopped by user."); return
                
                data = classified_results.get(image_path, {})
                confidence = data.get("confidence", 0.0)

                action = self._prompt_for_image_review(image_path, "not_info", confidence, title_suffix="Review 'not_info' Image")
                if action == "change_to_info":
                    images_to_change_to_info.append(image_path)
                    self._log(f"User re-classified {os.path.basename(image_path)} to 'info'.")
            
            if images_to_change_to_info:
                not_info_images = [p for p in not_info_images if p not in images_to_change_to_info]
                info_images.extend(images_to_change_to_info)

        # --- Process all 'info' images (original + re-classified) ---
        manual_processed_info = {}
        for image_path in info_images:
            if not self.is_organizing: self._log("Organization stopped by user."); return
            self._log(f"Showing manual dialog for {os.path.basename(image_path)}")
            
            patient_name, patient_id, classified_as_clinical_by_user = self.get_patient_info(image_path)
            
            if classified_as_clinical_by_user:
                not_info_images.append(image_path)
                self._log(f"User re-classified {os.path.basename(image_path)} as clinical.")
            elif patient_name and patient_id:
                manual_processed_info[image_path] = (patient_name, patient_id)
            else:
                self._log(f"Skipping organization for {os.path.basename(image_path)} due to missing patient info.")

        # --- Organize all files based on final classifications ---
        patient_info_count = len(manual_processed_info)
        clinical_count = len(not_info_images)
        current_patient_folder = None
        all_images_sorted = sorted(all_image_files, key=lambda p: os.path.getmtime(p))

        for image_path in all_images_sorted:
            if not self.is_organizing: self._log("Organization stopped by user."); return

            try:
                image = PIL.Image.open(image_path)
                exif_data = image._getexif()
                creation_datetime = datetime.strptime(exif_data[36867], '%Y:%m:%d %H:%M:%S') if exif_data and 36867 in exif_data else datetime.fromtimestamp(os.path.getmtime(image_path))
            except Exception:
                creation_datetime = datetime.fromtimestamp(os.path.getmtime(image_path))
            
            formatted_date = creation_datetime.strftime('%Y-%m-%d')
            base_name, ext = os.path.splitext(os.path.basename(image_path))
            new_filename = f"{formatted_date} {base_name}{ext}"

            if image_path in manual_processed_info:
                patient_name, patient_id = manual_processed_info[image_path]
                folder_name_patient_id = f"{patient_name} {patient_id}"
                patient_folder_dest = os.path.join(dest_dir, folder_name_patient_id)

                week_day = get_korean_weekday(creation_datetime.weekday())
                year_folder = creation_datetime.strftime('%Y년도')
                month_folder = creation_datetime.strftime('%m월')
                day_folder = creation_datetime.strftime(f'%Y.%m.%d ({week_day})')
                target_date_folder = os.path.join(date_dest_dir, year_folder, month_folder, day_folder, organizer_name)
                target_patient_folder_in_date = os.path.join(target_date_folder, folder_name_patient_id)

                self._copy_image_to_destinations(image_path, new_filename, patient_folder_dest, target_patient_folder_in_date)
                current_patient_folder = target_patient_folder_in_date

            elif image_path in not_info_images:
                if current_patient_folder:
                    path_parts = current_patient_folder.split(os.sep)
                    folder_name_patient_id = path_parts[-1]
                    patient_folder_dest = os.path.join(dest_dir, folder_name_patient_id)
                    self._copy_image_to_destinations(image_path, new_filename, patient_folder_dest, current_patient_folder)
                else:
                    unclassified_folder_dest = os.path.join(dest_dir, "unclassified_clinical")
                    unclassified_folder_date_dest = os.path.join(date_dest_dir, "unclassified_clinical")
                    self._copy_image_to_destinations(image_path, new_filename, unclassified_folder_dest, unclassified_folder_date_dest)

        end_time = time.time()
        elapsed_time = end_time - start_time
        summary_message = (
            f"\n--- Organization Complete ---\n"
            f"Total elapsed time: {elapsed_time:.2f} seconds\n"
            f"Patient info files processed: {patient_info_count}\n"
            f"Clinical files processed: {clinical_count}\n"
            f"-----------------------------\n"
        )
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

        # Style for the Combobox
        style = ttk.Style(dialog)
        style.configure("Blue.TCombobox", foreground="blue")
        # The default style is "TCombobox", which will have the default foreground (black)

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
        id_entry = ttk.Combobox(dialog, width=38, style="TCombobox") # Start with default style
        id_entry.pack()

        result = {"name": None, "id": None, "classified_as_clinical": False}
        id_selection_confirmed = False

        def _confirm_id_selection(event=None):
            nonlocal id_selection_confirmed
            id_selection_confirmed = True
            id_entry.configure(style="TCombobox") # Reset to default style on interaction

        def _clear_name_entry(_event=None):
            name_entry.delete(0, tk.END)
            name_entry.unbind("<Button-1>")

        def _reset_name_color(_event=None):
            name_entry.config(fg="black")
            name_entry.unbind("<KeyPress>")

        def _autocomplete_id(_event=None):
            nonlocal id_selection_confirmed
            id_selection_confirmed = False # Reset confirmation on new autocomplete
            id_entry.configure(style="TCombobox") # Reset style to default

            name_typed = name_entry.get().strip()
            id_entry.set('')
            id_entry.config(values=[])

            if not name_typed: return

            matching_patients = [p for p in self._get_existing_patient_info() if p["name"].lower() == name_typed.lower()]

            if matching_patients:
                ids = [p["id"] for p in matching_patients]
                id_entry.config(values=ids)
                id_entry.set(ids[0])
                id_entry.configure(style="Blue.TCombobox") # Apply blue style
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
            
            # The user has clicked OK, so we now validate the current state of the form.
            name = name_entry.get().strip()
            patient_id = id_entry.get().strip()

            # First, check for confirmation if multiple options were present
            values_in_list = id_entry.cget('values')
            if isinstance(values_in_list, str): # Tkinter can return a string representation
                values_in_list = dialog.tk.splitlist(values_in_list)

            if len(values_in_list) > 1 and not id_selection_confirmed:
                messagebox.showwarning("Confirm Selection", 
                                       "동일한 이름의 환자가 여러 명 있습니다. ID를 확인하거나 목록에서 선택한 후 다시 시도해주세요.", 
                                       parent=dialog)
                return

            # Then, check for swapped or empty inputs
            if name.isdigit() and not patient_id.isdigit():
                self._log(f"Swapped input detected. Correcting: Name='{patient_id}', ID='{name}'")
                name, patient_id = patient_id, name

            if not name or not patient_id:
                messagebox.showwarning("Input Error", "Patient Name and ID cannot be empty.", parent=dialog)
                return

            # If all checks pass, set the result and close the dialog
            result["name"], result["id"] = name, patient_id
            dialog.destroy()

        def on_classify_as_clinical():
            result["classified_as_clinical"] = True
            dialog.destroy()

        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Classify as Clinical", command=on_classify_as_clinical).pack(side=tk.RIGHT, padx=10)

        # Bindings
        name_entry.bind("<Button-1>", _clear_name_entry)
        name_entry.bind("<FocusOut>", _autocomplete_id)
        name_entry.bind("<Return>", lambda e: id_entry.focus_set())
        
        id_entry.bind("<FocusOut>", _autocomplete_name)
        id_entry.bind("<Return>", lambda e: on_ok())
        id_entry.bind("<<ComboboxSelected>>", _confirm_id_selection)
        id_entry.bind("<KeyPress>", _confirm_id_selection)

        self.root.wait_window(dialog)

        # Post-dialog logic (unchanged)
        if not result["classified_as_clinical"] and result["name"] and result["id"]:
            existing_patients = self._get_existing_patient_info()
            exact_match_found = any(p["name"].lower() == result["name"].lower() and p["id"].lower() == result["id"].lower() for p in existing_patients)

            if not exact_match_found:
                self._add_patient_to_cache(result["name"], result["id"], f"{result['name']} {result['id']}")
                similar_suggestions = self._find_similar_patients(result["name"], result["id"], existing_patients)
                if similar_suggestions:
                    if threading.current_thread() != threading.main_thread():
                        q = queue.Queue()
                        self.root.after(0, lambda: q.put(self._prompt_for_suggestion(similar_suggestions, result["name"], result["id"])))
                        chosen_name, chosen_id = q.get()
                    else:
                        chosen_name, chosen_id = self._prompt_for_suggestion(similar_suggestions, result["name"], result["id"])
                    result["name"], result["id"] = chosen_name, chosen_id

        return result["name"], result["id"], result["classified_as_clinical"]

    def _prompt_for_image_review(self, image_path, predicted_class, confidence, title_suffix="", message=None):
        if threading.current_thread() != threading.main_thread():
            q = queue.Queue()
            self.root.after(0, lambda: q.put(self._show_image_review_dialog(image_path, predicted_class, confidence, title_suffix, message)))
            return q.get()
        else:
            return self._show_image_review_dialog(image_path, predicted_class, confidence, title_suffix, message)

    def _show_image_review_dialog(self, image_path, predicted_class, confidence, title_suffix, message):
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Image Review: {title_suffix}")
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
        if message:
            info_text += f"\n\n{message}"
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

    def _prompt_for_same_name_selection(self, new_name, new_id, existing_patients):
        dialog = tk.Toplevel(self.root)
        dialog.title("Confirm Patient")
        dialog.geometry(f"500x350+{self.root.winfo_x()+200}+{self.root.winfo_y()+200}")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text=f"A patient named '{new_name}' already exists with a different ID.\nPlease choose an action:", justify=tk.LEFT).pack(pady=10, padx=10)

        selected_patient_info = tk.StringVar()
        result = {"action": None, "patient": None}

        new_patient_text = f"Create a new patient: {new_name} {new_id}"
        tk.Radiobutton(dialog, text=new_patient_text, variable=selected_patient_info, value="create_new").pack(anchor="w", padx=20)

        tk.Label(dialog, text="Or select an existing patient:").pack(anchor="w", padx=15, pady=(10,0))
        for patient in existing_patients:
            text = f"{patient['name']} {patient['id']}"
            tk.Radiobutton(dialog, text=text, variable=selected_patient_info, value=str(patient)).pack(anchor="w", padx=20)
        
        selected_patient_info.set("create_new")

        def on_confirm():
            selection = selected_patient_info.get()
            if selection == "create_new":
                result["action"] = "create_new"
            else:
                result["action"] = "select_existing"
                result["patient"] = eval(selection)
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="Confirm", command=on_confirm).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=10)

        self.root.wait_window(dialog)
        return result["action"], result["patient"]

    def _prompt_for_suggestion(self, suggestions, original_name, original_id):
        suggestion_window = tk.Toplevel(self.root)
        suggestion_window.title("Did you mean...?",)
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

    def synchronize_folders(self):
        self._log("--- Starting Fast Folder Synchronization (New/Modified Files Only) ---")

        dest_dir = self.dest_dir_entry.get()
        date_dest_dir = self.date_dest_dir_entry.get()
        organizer_name = self.organizer_name_entry.get().strip()

        if not all([os.path.isdir(dest_dir), os.path.isdir(date_dest_dir), organizer_name]):
            self._log("Error: Sync requires valid Dest, Date Dest, and Organizer Name.")
            return

        # Use the last sync timestamp to only check recent files.
        # Default to 0.0 to force a full scan on the first ever run.
        last_sync_ts = self.last_sync_timestamp

        self._log(f"Scanning for files modified since {datetime.fromtimestamp(last_sync_ts).strftime('%Y-%m-%d %H:%M:%S')}")
        
        copied_file_count = 0
        date_regex = re.compile(r"^(\d{4}-\d{2}-\d{2})")

        for patient_folder_name in os.listdir(dest_dir):
            patient_folder_path = os.path.join(dest_dir, patient_folder_name)
            if not os.path.isdir(patient_folder_path): continue

            for filename in os.listdir(patient_folder_path):
                source_file_path = os.path.join(patient_folder_path, filename)
                
                # Skip non-files and check modification time first for efficiency
                if not os.path.isfile(source_file_path) or os.path.getmtime(source_file_path) <= last_sync_ts:
                    continue

                # File is new or modified, proceed with copying
                match = date_regex.search(filename)
                if not match: continue

                file_date_str = match.group(1)
                try:
                    file_dt = datetime.strptime(file_date_str, '%Y-%m-%d')
                    week_day = get_korean_weekday(file_dt.weekday())
                    
                    year_folder = file_dt.strftime('%Y년도')
                    month_folder = file_dt.strftime('%m월')
                    day_folder = file_dt.strftime(f'%Y.%m.%d ({week_day})')
                    
                    target_patient_folder_in_date = os.path.join(
                        date_dest_dir,
                        year_folder,
                        month_folder,
                        day_folder,
                        organizer_name,
                        patient_folder_name
                    )
                    
                    # Use the existing copy function
                    self._copy_image_to_single_destination(source_file_path, filename, target_patient_folder_in_date)
                    copied_file_count += 1

                except ValueError:
                    self._log(f"Skipping file with invalid date format in name: {filename}")
                    continue
        
        self._log(f"--- Fast Sync Complete. Copied {copied_file_count} new/modified files. ---")
        
        # Update the timestamp for the next run
        self.last_sync_timestamp = datetime.now().timestamp()
        self.save_paths()
        self._log("--- Folder Synchronization Finished ---")

    def _copy_image_to_single_destination(self, source_path, new_filename, dest_folder):
        try:
            os.makedirs(dest_folder, exist_ok=True)
            dest_path = os.path.join(dest_folder, new_filename)
            shutil.copy2(source_path, dest_path)
            self._log(f"Copied {os.path.basename(source_path)} to {dest_path}")
        except Exception as e:
            self._log(f"Error copying {os.path.basename(source_path)} to {dest_folder}: {e}")

    def _copy_image_to_destinations(self, source_path, new_filename, dest1, dest2):
        self._copy_image_to_single_destination(source_path, new_filename, dest1)
        self._copy_image_to_single_destination(source_path, new_filename, dest2)

if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoOrganizerApp(root)
    root.mainloop()
