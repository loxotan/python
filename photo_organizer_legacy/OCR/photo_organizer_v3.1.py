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
from tkinter import filedialog, messagebox, simpledialog
import os
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
import pytesseract
import cv2
import numpy as np


# Tesseract OCR 경로 설정 (동적으로)
def set_tesseract_path():
    print("Setting Tesseract path...")
    try:
        if getattr(sys, 'frozen', False): # PyInstaller로 빌드된 경우
            base_path = sys._MEIPASS
            tesseract_path = os.path.join(base_path, 'Tesseract-OCR', 'tesseract.exe')
            print(f"Frozen app detected. Base path (_MEIPASS): {base_path}")
            print(f"Tesseract path constructed: {tesseract_path}")
        else: # 일반 파이썬 스크립트로 실행된 경우
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            print(f"Running as script. Tesseract path: {tesseract_path}")

        print(f"Checking if Tesseract executable exists at: {tesseract_path}")
        if not os.path.exists(tesseract_path):
            print("--- TESSERACT NOT FOUND ---")
            # 디버깅을 위해 예상 경로의 내용물을 나열
            if getattr(sys, 'frozen', False):
                tesseract_dir = os.path.join(base_path, 'Tesseract-OCR')
                print(f"Checking contents of bundled directory: {tesseract_dir}")
                if os.path.exists(tesseract_dir):
                    print(f"Contents: {os.listdir(tesseract_dir)}")
                else:
                    print("Tesseract-OCR directory does not exist in the bundle.")
                    print(f"Contents of base_path ({base_path}): {os.listdir(base_path)}")
            raise FileNotFoundError(f"Tesseract executable not found at the specified path: {tesseract_path}")
        
        print("Tesseract executable found. Assigning path to pytesseract.")
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print("Tesseract path set successfully.")
    except Exception as e:
        print("--- CRITICAL ERROR IN set_tesseract_path ---")
        print(f"Error: {e}")
        print(traceback.format_exc())
        raise # 예외를 다시 발생시켜 프로그램 중단
        print(f"Running as script. Tesseract path: {tesseract_path}")
    
    # 이 라인이 if/else 블록 밖에 있어야 두 경우 모두에 적용됩니다.
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print("Tesseract path set.")

set_tesseract_path()

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

        self.use_ocr_var = tk.BooleanVar(value=True)
        self.use_ocr_checkbox = tk.Checkbutton(self.control_frame, text="환자 정보 자동인식 (OCR) 사용", variable=self.use_ocr_var)
        self.use_ocr_checkbox.pack(pady=1)

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
        self.root.protocol("WM_DELETE_WINDOW", self.save_paths)

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
            id_match = re.search(r'(\b\d{4,})\b', folder_name)
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

        # Re-check stop flag after counting images, before processing
        if not self.is_organizing:
            self._log("Organization stopped by user.")
            return

        # Store classification results
        classified_results = {} # {image_path: "info" or "not_info"}

        if self.model is None:
            self._log("Deep learning model not loaded. Aborting classification.")
            return

        self.model.eval() # Set model to evaluation mode

        self._log("Starting image classification loop.")
        for image_path in all_image_files:
            if not self.is_organizing: # Check stop flag
                self._log("Organization stopped by user.")
                return
            try:
                image = PIL.Image.open(image_path).convert('RGB')
                input_tensor = preprocess(image)
                input_batch = input_tensor.unsqueeze(0)

                with torch.no_grad():
                    input_batch = input_batch.to(DEVICE)
                    output = self.model(input_batch)
                    confidence = torch.sigmoid(output).item()
                    prediction = confidence > 0.5
                    predicted_class = "not_info" if prediction else "info"

                classified_results[image_path] = {"class": predicted_class, "confidence": confidence}
                self._log(f"Classified {os.path.basename(image_path)} as {predicted_class} (Confidence: {confidence:.2f})")

            except Exception as e:
                self._log(f"Error classifying image {os.path.basename(image_path)}: {e}")
                classified_results[image_path] = {"class": "not_info", "confidence": 0.0} # Default to clinical if classification fails

        self._log("Image classification loop finished. Starting organization based on results.")

        # --- New Organizing Logic ---
        # 1. Separate info images from clinical images
        info_images = [p for p, data in classified_results.items() if data["class"] == "info"]
        not_info_images = [p for p, data in classified_results.items() if data["class"] == "not_info"]

        # 2. Automatically process all info images first
        auto_processed_info = {} # {image_path: (name, id)}
        manual_review_needed = [] # [image_path]

        use_ocr = self.use_ocr_var.get()

        if use_ocr:
            self._log(f"\n--- Starting Automatic OCR Phase for {len(info_images)} images ---\n")

            # Use ThreadPoolExecutor for parallel OCR processing
            # Use a number of workers appropriate for your CPU, e.g., os.cpu_count()
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                # Create a future for each OCR task
                future_to_image = {executor.submit(self._perform_ocr_and_db_lookup, image_path): image_path for image_path in info_images}

                for future in as_completed(future_to_image):
                    image_path = future_to_image[future]
                    try:
                        name, patient_id = future.result()
                        if name and patient_id:
                            auto_processed_info[image_path] = (name, patient_id)
                        else:
                            manual_review_needed.append(image_path)
                    except Exception as exc:
                        self._log(f'{image_path} generated an exception: {exc}')
                        manual_review_needed.append(image_path)

            self._log(f"--- Automatic OCR Phase Finished. {len(auto_processed_info)} images automatically identified. ---")
        else:
            self._log("\n--- OCR is disabled. All info images will require manual review. ---\n")
            manual_review_needed = info_images # If OCR is off, all info images need manual review

        # 3. Manually process images that need review
        manual_processed_info = {}
        for image_path in manual_review_needed:
            if not self.is_organizing: return
            self._log(f"Showing manual dialog for {os.path.basename(image_path)}")
            
            patient_name, patient_id, classified_as_clinical_by_user = self.get_patient_info(image_path)
            
            if classified_as_clinical_by_user:
                not_info_images.append(image_path) # Move to the other list
                self._log(f"User re-classified {os.path.basename(image_path)} as clinical.")
            elif patient_name and patient_id:
                manual_processed_info[image_path] = (patient_name, patient_id)
            else:
                self._log(f"Skipping organization for {os.path.basename(image_path)} due to missing patient info in manual review.")

        # 4. Combine all processed info and organize all files
        all_processed_info = {**auto_processed_info, **manual_processed_info}
        patient_info_count = len(all_processed_info)
        clinical_count = len(not_info_images)
        current_patient_folder = None

        # Create a sorted list of all images to process them in order
        all_images_sorted = sorted(all_image_files, key=lambda p: os.path.getmtime(p))

        for image_path in all_images_sorted:
            if not self.is_organizing: return

            # Get creation time and new filename
            try:
                image = PIL.Image.open(image_path)
                exif_data = image._getexif()
                if exif_data and 36867 in exif_data:
                    creation_datetime = datetime.strptime(exif_data[36867], '%Y:%m:%d %H:%M:%S')
                else:
                    creation_datetime = datetime.fromtimestamp(os.path.getmtime(image_path))
            except Exception as e:
                creation_datetime = datetime.fromtimestamp(os.path.getmtime(image_path))
            
            formatted_date = creation_datetime.strftime('%Y-%m-%d')
            base_name, ext = os.path.splitext(os.path.basename(image_path))
            new_filename = f"{formatted_date} {base_name}{ext}"

            # --- Organize based on final classification ---
            if image_path in all_processed_info:
                # This is an info file
                patient_name, patient_id = all_processed_info[image_path]
                
                folder_name_patient_id = f"{patient_name} {patient_id}"
                patient_folder_dest = os.path.join(dest_dir, folder_name_patient_id)
                os.makedirs(patient_folder_dest, exist_ok=True)

                week_day = get_korean_weekday(creation_datetime.weekday())
                date_folder_name = creation_datetime.strftime(f'%Y년도')
                month_folder_name = creation_datetime.strftime(f'%m월')
                day_folder_name = creation_datetime.strftime(f'%Y.%m.%d ({week_day})')

                target_date_folder = os.path.join(date_dest_dir, date_folder_name, month_folder_name, day_folder_name, organizer_name)
                os.makedirs(target_date_folder, exist_ok=True)

                target_patient_folder_in_date = os.path.join(target_date_folder, folder_name_patient_id)
                os.makedirs(target_patient_folder_in_date, exist_ok=True)

                self._copy_image_to_destinations(image_path, new_filename, patient_folder_dest, target_patient_folder_in_date)
                current_patient_folder = target_patient_folder_in_date

            elif image_path in not_info_images:
                # This is a clinical file
                if current_patient_folder:
                    path_parts = current_patient_folder.split(os.sep)
                    folder_name_patient_id = path_parts[-1]
                    self._copy_image_to_single_destination(image_path, new_filename, current_patient_folder)
                    patient_folder_dest = os.path.join(dest_dir, folder_name_patient_id)
                    self._copy_image_to_single_destination(image_path, new_filename, patient_folder_dest)
                else:
                    unclassified_folder_dest = os.path.join(dest_dir, "unclassified_clinical")
                    self._copy_image_to_single_destination(image_path, new_filename, unclassified_folder_dest)
                    unclassified_folder_date_dest = os.path.join(date_dest_dir, "unclassified_clinical")
                    self._copy_image_to_single_destination(image_path, new_filename, unclassified_folder_date_dest)


        # --- Final Summary ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        summary_message = (
            f"\n--- Organization Complete ---\n"
            f"Total elapsed time: {elapsed_time:.2f} seconds\n"
            f"Patient info files processed: {patient_info_count}\n"
            f"  - Automatically by OCR: {len(auto_processed_info)}\n"
            f"  - Manually entered: {len(manual_processed_info)}\n"
            f"Clinical files processed: {clinical_count}\n"
            f"-----------------------------\n"
        )
        self._log(summary_message)

    def _perform_ocr_and_db_lookup(self, image_path):
        """
        Performs OCR on an image and looks up the found ID in the database.
        This function does NOT create any UI elements.
        Returns (name, id) if a match is found, otherwise (None, None).
        """
        try:
            # --- Enhanced Image Preprocessing for OCR ---
            img_pil = PIL.Image.open(image_path).convert('RGB')
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # 1. Resize for better OCR performance (upscaling can help)
            scale_factor = 2
            width = int(img_cv.shape[1] * scale_factor)
            height = int(img_cv.shape[0] * scale_factor)
            resized_img = cv2.resize(img_cv, (width, height), interpolation=cv2.INTER_CUBIC)

            # 2. Convert to grayscale
            gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

            # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(gray)

            # 4. Apply Adaptive Thresholding
            binary_img = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                               cv2.THRESH_BINARY, 11, 2)

            # --- OCR Logic ---
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            ocr_text = ""
            filename = os.path.basename(image_path)
            try:
                # Set a 3-second timeout for the OCR process
                ocr_text = pytesseract.image_to_string(binary_img, lang='kor', config=custom_config, timeout=3)
            except RuntimeError as timeout_error:
                # This exception is raised by pytesseract on timeout
                self._log(f"OCR timed out for {filename} after 3 seconds. Sending to manual review.")
                return None, None # Give up on this image

            id_match = re.search(r'(\b\d{4,7}\b)', ocr_text)
            
            if id_match:
                potential_id = id_match.group(1).strip()
                existing_patients = self._get_existing_patient_info()
                
                # Find patient with matching ID
                matched_patient = next((p for p in existing_patients if p["id"] == potential_id), None)

                if matched_patient:
                    self._log(f"Recognized {filename} as {matched_patient['name']} {matched_patient['id']}")
                    return matched_patient["name"], matched_patient["id"]
                else:
                    self._log(f"OCR found ID {potential_id} in {filename}, but no matching patient in database.")
                    raw_text = ocr_text.strip()
                    return None, None
            else:
                raw_text = ocr_text.strip()
                self._log(f"OCR did not find a patient ID in {filename}. Raw text: '{raw_text}'")

        except Exception as e:
            self._log(f"Auto OCR failed for {os.path.basename(image_path)}: {e}")
        
        return None, None

    def get_patient_info(self, image_path):
        # If called from a worker thread, delegate to the main thread and wait for the result.
        if threading.current_thread() != threading.main_thread():
            q = queue.Queue()
            self.root.after(0, lambda: q.put(self._show_patient_info_dialog(image_path)))
            return q.get() # This will block the worker thread until the main thread puts the result in the queue
        # If called from the main thread, just show the dialog directly
        return self._show_patient_info_dialog(image_path)

    def _show_patient_info_dialog(self, image_path):
        dialog = tk.Toplevel(self.root)
        dialog.title("Enter Patient Information")
        dialog.geometry(f"500x620+{self.root.winfo_x()+150}+{self.root.winfo_y()+150}")
        dialog.transient(self.root)
        dialog.grab_set()

        id_match = None # Initialize id_match

        use_ocr = self.use_ocr_var.get()

        try:
            # --- Enhanced Image Preprocessing for OCR ---
            img_pil = PIL.Image.open(image_path).convert('RGB')
            if use_ocr:
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                # 1. Resize for better OCR performance (upscaling can help)
                scale_factor = 2
                width = int(img_cv.shape[1] * scale_factor)
                height = int(img_cv.shape[0] * scale_factor)
                resized_img = cv2.resize(img_cv, (width, height), interpolation=cv2.INTER_CUBIC)

                # 2. Convert to grayscale
                gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

                # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                contrast_enhanced = clahe.apply(gray)

                # 4. Apply Adaptive Thresholding
                binary_img = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                   cv2.THRESH_BINARY, 11, 2)

                # --- OCR Logic ---
                try:
                    # Use PSM 11 for more flexible text detection
                    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
                    # Set a 3-second timeout for the OCR process
                    ocr_text = pytesseract.image_to_string(binary_img, lang='kor', config=custom_config, timeout=3)
                    self._log(f"OCR Result for {os.path.basename(image_path)}: '{ocr_text.strip()}'")

                    # Regex to find a 4 to 7 digit number (patient ID)
                    id_match = re.search(r'(\b\d{4,7}\b)', ocr_text)
                    if id_match:
                        potential_id = id_match.group(1).strip()
                        
                        # Check if this ID exists in our database
                        existing_patients = self._get_existing_patient_info()
                        for patient in existing_patients:
                            if patient["id"] == potential_id:
                                # --- MATCH FOUND - SKIP DIALOG ---
                                self._log(f"Automatic match found for ID {potential_id}: {patient['name']}. Skipping dialog.")
                                return patient["name"], patient["id"], False # Return the matched info directly

                except RuntimeError as timeout_error:
                    # This exception is raised by pytesseract on timeout
                    self._log(f"OCR timed out for {os.path.basename(image_path)} during manual review setup.")
                except Exception as e:
                    self._log(f"OCR failed for {os.path.basename(image_path)}: {e}")
                # --- End OCR Logic ---

            # Display the original image (before preprocessing) in the dialog
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
        id_entry = tk.Entry(dialog, width=40)
        id_entry.pack()

        # --- Insert OCR results (if any, and no auto-match was found) ---
        if id_match:
             id_entry.insert(0, id_match.group(1).strip())
        # ---

        result = {"name": None, "id": None, "classified_as_clinical": False}

        # --- Autocomplete Logic ---
        def _clear_name_entry(_event=None):
            name_entry.delete(0, tk.END)
            name_entry.unbind("<Button-1>") # Unbind after first click
        def _reset_id_color(_event=None):
            id_entry.config(fg="black")
            id_entry.unbind("<KeyPress>")

        def _reset_name_color(_event=None):
            name_entry.config(fg="black")
            name_entry.unbind("<KeyPress>")

        def _autocomplete_id(_event=None):
            name_typed = name_entry.get().strip()
            dest_dir = self.dest_dir_entry.get()
            if not name_typed or not os.path.isdir(dest_dir) or id_entry.get():
                return

            existing_patients = self._get_existing_patient_info()
            for patient in existing_patients:
                if patient["name"].lower().startswith(name_typed.lower()):
                    id_entry.delete(0, tk.END)
                    id_entry.insert(0, patient["id"])
                    id_entry.config(fg="blue")
                    id_entry.bind("<KeyPress>", _reset_id_color)
                    break

        def _autocomplete_name(_event=None):
            id_typed = id_entry.get().strip()
            dest_dir = self.dest_dir_entry.get()
            if not id_typed or not os.path.isdir(dest_dir) or name_entry.get():
                return

            existing_patients = self._get_existing_patient_info()
            for patient in existing_patients:
                if patient["id"].lower() == id_typed.lower():
                    name_entry.delete(0, tk.END)
                    name_entry.insert(0, patient["name"])
                    name_entry.config(fg="blue")
                    name_entry.bind("<KeyPress>", _reset_name_color)
                    break
        # ---

        def on_ok():
            # Run autocomplete one last time before submitting, in case FocusOut was not triggered
            if not name_entry.get() and id_entry.get():
                _autocomplete_name()
            if not id_entry.get() and name_entry.get():
                _autocomplete_id()

            name = name_entry.get().strip()
            patient_id = id_entry.get().strip()

            # --- Heuristic to detect and correct swapped Name and ID ---
            # If 'name' looks like a number and 'id' does not, swap them.
            if name.isdigit() and not patient_id.isdigit():
                self._log(f"Swapped input detected. Correcting: Name='{patient_id}', ID='{name}'")
                name, patient_id = patient_id, name # Swap them

            if not name or not patient_id:
                messagebox.showwarning("Input Error", "Patient Name and ID cannot be empty.", parent=dialog)
                return

            # --- Improved Same Name Check ---
            existing_patients = self._get_existing_patient_info()
            patients_with_same_name = [p for p in existing_patients if p['name'] == name and p['id'] != patient_id]
            
            # Check if the exact same name and ID already exists
            exact_match_exists = any(p['name'] == name and p['id'] == patient_id for p in existing_patients)

            if patients_with_same_name and not exact_match_exists:
                # This is a new ID for an existing name. Ask the user to confirm.
                chosen_action, chosen_patient = self._prompt_for_same_name_selection(name, patient_id, patients_with_same_name)
                
                if chosen_action == "select_existing":
                    result["name"] = chosen_patient['name']
                    result["id"] = chosen_patient['id']
                elif chosen_action == "create_new":
                    result["name"] = name
                    result["id"] = patient_id
                else: # User cancelled
                    return 
            else:
                # No conflicts, or an exact match already exists. Proceed.
                result["name"] = name
                result["id"] = patient_id
            # ---

            dialog.destroy()

        def on_classify_as_clinical():
            result["classified_as_clinical"] = True
            dialog.destroy()

        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Classify as Clinical", command=on_classify_as_clinical).pack(side=tk.RIGHT, padx=10)

        # Bind events
        name_entry.bind("<Button-1>", _clear_name_entry)
        name_entry.bind("<FocusOut>", _autocomplete_id)
        name_entry.bind("<Return>", lambda e: id_entry.focus_set())
        id_entry.bind("<FocusOut>", _autocomplete_name)
        id_entry.bind("<Return>", lambda e: on_ok())

        self.root.wait_window(dialog)

        # --- Post-dialog processing for suggestions (similar to v1.9) ---
        if not result["classified_as_clinical"] and result["name"] and result["id"]:
            dest_dir = self.dest_dir_entry.get()
            existing_patients = self._get_existing_patient_info()
            
            exact_match_found = any(
                p["name"].lower() == result["name"].lower() and 
                p["id"].lower() == result["id"].lower() 
                for p in existing_patients
            )

            if not exact_match_found:
                # Add the new patient to the cache before checking for suggestions
                self._add_patient_to_cache(result["name"], result["id"], f"{result['name']} {result['id']}")

                similar_suggestions = self._find_similar_patients(result["name"], result["id"], existing_patients)
                if similar_suggestions:
                    # This needs to be thread-safe as well
                    if threading.current_thread() != threading.main_thread():
                        q = queue.Queue()
                        self.root.after(0, lambda: q.put(self._prompt_for_suggestion(similar_suggestions, result["name"], result["id"])))
                        chosen_name, chosen_id = q.get()
                    else:
                        chosen_name, chosen_id = self._prompt_for_suggestion(similar_suggestions, result["name"], result["id"])
                    result["name"] = chosen_name
                    result["id"] = chosen_id

        return result["name"], result["id"], result["classified_as_clinical"]

    

    

    

    def _prompt_for_same_name_selection(self, new_name, new_id, existing_patients):
        dialog = tk.Toplevel(self.root)
        dialog.title("Confirm Patient")
        dialog.geometry(f"500x350+{self.root.winfo_x()+200}+{self.root.winfo_y()+200}")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text=f"A patient named '{new_name}' already exists with a different ID.\nPlease choose an action:", justify=tk.LEFT).pack(pady=10, padx=10)

        selected_patient_info = tk.StringVar()
        result = {"action": None, "patient": None}

        # ---
        # Option to create a new patient
        new_patient_text = f"Create a new patient: {new_name} {new_id}"
        tk.Radiobutton(dialog, text=new_patient_text, variable=selected_patient_info, value="create_new").pack(anchor="w", padx=20)

        # Options for existing patients
        tk.Label(dialog, text="Or select an existing patient:").pack(anchor="w", padx=15, pady=(10,0))
        for patient in existing_patients:
            text = f"{patient['name']} {patient['id']}"
            # Store the patient dict as a string and retrieve it later
            tk.Radiobutton(dialog, text=text, variable=selected_patient_info, value=str(patient)).pack(anchor="w", padx=20)
        
        selected_patient_info.set("create_new") # Default to creating a new patient

        def on_confirm():
            selection = selected_patient_info.get()
            if selection == "create_new":
                result["action"] = "create_new"
            else:
                # Convert the string back to a dictionary
                result["action"] = "select_existing"
                result["patient"] = eval(selection) 
            dialog.destroy()

        def on_cancel():
            dialog.destroy() # This will result in returning the default None values

        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="Confirm", command=on_confirm).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=10)

        self.root.wait_window(dialog)
        return result["action"], result["patient"]

    def _prompt_for_suggestion(self, suggestions, original_name, original_id):
        suggestion_window = tk.Toplevel(self.root)
        suggestion_window.title("Did you mean...?")
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
        self._log("--- Starting Folder Synchronization ---")

        dest_dir = self.dest_dir_entry.get()
        date_dest_dir = self.date_dest_dir_entry.get()
        organizer_name = self.organizer_name_entry.get().strip()

        if not all([os.path.isdir(dest_dir), os.path.isdir(date_dest_dir), organizer_name]):
            self._log("Error: Sync requires valid Dest, Date Dest, and Organizer Name.")
            return

        date_regex = re.compile(r"^(\d{4}-\d{2}-\d{2})")
        
        # --- Phase 1: Build a set of expected files in the date directory ---
        expected_date_files = set()
        source_of_truth_files = {} 

        self._log("Scanning patient folders (source of truth)...")
        for patient_folder_name in os.listdir(dest_dir):
            patient_folder_path = os.path.join(dest_dir, patient_folder_name)
            if not os.path.isdir(patient_folder_path):
                continue

            for filename in os.listdir(patient_folder_path):
                source_file_path = os.path.join(patient_folder_path, filename)
                if not os.path.isfile(source_file_path):
                    continue

                match = date_regex.search(filename)
                if not match:
                    continue

                file_date_str = match.group(1)
                try:
                    file_dt = datetime.strptime(file_date_str, '%Y-%m-%d')
                    week_day = get_korean_weekday(file_dt.weekday())
                    
                    expected_path = os.path.join(
                        date_dest_dir,
                        file_dt.strftime(f'%Y년도'),
                        file_dt.strftime(f'%m월'),
                        file_dt.strftime(f'%Y.%m.%d ({week_day})'),
                        organizer_name,
                        patient_folder_name,
                        filename
                    )
                    expected_date_files.add(expected_path)
                    source_of_truth_files[expected_path] = source_file_path

                except ValueError:
                    self._log(f"Skipping file with invalid date format in name: {filename}")
                    continue
        
        self._log(f"Source scan complete. Found {len(expected_date_files)} files as the source of truth for organizer '{organizer_name}'.")

        # --- Phase 2: Scan the date directory for this organizer and find discrepancies ---
        self._log(f"Scanning date-based folders for comparison (organizer: {organizer_name})...")
        actual_date_files = set()
        for root, dirs, files in os.walk(date_dest_dir):
            # Only look in paths that are part of the current organizer's potential folder structure
            if organizer_name in root.split(os.sep):
                # Don't touch the orphaned files directory during the scan
                if '_orphaned_files' in dirs:
                    dirs.remove('_orphaned_files') # prevent os.walk from going into it

                for filename in files:
                    actual_date_files.add(os.path.join(root, filename))

        # --- Phase 3: Correct the discrepancies ---
        files_to_copy = expected_date_files - actual_date_files
        files_to_remove = actual_date_files - expected_date_files

        self._log(f"Sync check complete. Files to copy: {len(files_to_copy)}. Files to remove: {len(files_to_remove)}.")

        # Perform copies
        if files_to_copy:
            self._log(f"--- Copying {len(files_to_copy)} missing files to date folder ---")
            for path in files_to_copy:
                source_path = source_of_truth_files.get(path)
                if source_path:
                    try:
                        dest_folder = os.path.dirname(path)
                        os.makedirs(dest_folder, exist_ok=True)
                        shutil.copy2(source_path, path)
                        self._log(f"COPIED: {os.path.basename(path)} to {dest_folder}")
                    except Exception as e:
                        self._log(f"ERROR copying {os.path.basename(path)}: {e}")
                else:
                    self._log(f"ERROR: Could not find source for expected file {path}")

        # Perform removals (by moving to an 'orphaned' directory with original structure)
        if files_to_remove:
            self._log(f"--- Moving {len(files_to_remove)} orphaned files from date folder ---")
            # Create a base folder for this sync run to avoid mixing files from different runs
            orphaned_dir_session = os.path.join(date_dest_dir, "_orphaned_files", f"sync_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(orphaned_dir_session, exist_ok=True)
            
            for path in files_to_remove:
                try:
                    # Recreate the original folder structure inside the orphaned directory
                    relative_path = os.path.relpath(path, date_dest_dir)
                    new_dest_path = os.path.join(orphaned_dir_session, relative_path)
                    
                    # Ensure the destination directory exists
                    os.makedirs(os.path.dirname(new_dest_path), exist_ok=True)
                    
                    shutil.move(path, new_dest_path)
                    self._log(f"MOVED (orphaned): {relative_path} to _orphaned_files folder")
                except Exception as e:
                    self._log(f"ERROR moving orphaned file {os.path.basename(path)}: {e}")

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
