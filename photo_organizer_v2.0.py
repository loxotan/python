import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import re
from datetime import datetime
import difflib

import torch
import torch.nn as nn
from torchvision import models, transforms
import PIL
import json

# --- 딥러닝 모델 설정 ---
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATHS_FILE = 'paths.json'

def build_model(num_classes):
    """MobileNetV2를 기반으로 한 전이 학습 모델을 구축합니다."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 1)
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

        self.start_button = tk.Button(self.button_frame, text="Start Organizing", command=self.start_organizing)
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_organizing, state=tk.DISABLED)
        self.exit_button = tk.Button(self.button_frame, text="Exit", command=self.on_exit)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=1)
        self.stop_button.pack(side=tk.RIGHT, padx=5, pady=1)
        self.exit_button.pack(side=tk.RIGHT, padx=5, pady=1)

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
            "제작자: 치주과 최수영"
        )
        tk.Label(self.model_info_frame, text=info_text, justify=tk.LEFT).pack(anchor="w", padx=5, pady=5)

        # Frame for logging
        self.log_frame = tk.Frame(root, padx=5, pady=2)
        self.log_frame.pack(padx=5, pady=2, fill="both", expand=True)

        self.log_label = tk.Label(self.log_frame, text="Log:")
        self.log_label.pack(anchor="w")

        # Create a frame for the Text widget and Scrollbar
        self.log_text_frame = tk.Frame(self.log_frame)
        self.log_text_frame.pack()

        self.log_text = tk.Text(self.log_text_frame, height=15, width=80)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.log_scrollbar = tk.Scrollbar(self.log_text_frame, command=self.log_text.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.log_scrollbar.set)

        # Load the deep learning model
        self.log_text.insert(tk.END, "Loading deep learning model...\n")
        self.log_text.see(tk.END)
        self.model = build_model(num_classes=2) # Assuming 2 classes: info, not_info
        model_path = 'photo_classifier_model.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.eval() # Set model to evaluation mode
            self.log_text.insert(tk.END, "Model loaded successfully.\n")
            self.log_text.see(tk.END)
        else:
            messagebox.showerror("Error", f"Model file '{model_path}' not found. Please ensure it's in the same directory.")
            self.log_text.insert(tk.END, f"Error: Model file '{model_path}' not found.\n")
            self.log_text.see(tk.END)
            self.model = None # Indicate that model loading failed

        # Load saved paths
        self.load_paths()

        # Save paths on window close
        self.root.protocol("WM_DELETE_WINDOW", self.save_paths)

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
                    if self.last_organizing_timestamp > 0:
                        self.last_organized_label.config(text=f"Last organized: {datetime.fromtimestamp(self.last_organizing_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        self.last_organized_label.config(text="Last organized: Never")
        except Exception as e:
            self.log_text.insert(tk.END, f"Error loading paths: {e}\n")
            self.log_text.see(tk.END)

    def save_paths(self):
        try:
            paths = {
                'source_dir': self.source_dir_entry.get(),
                'dest_dir': self.dest_dir_entry.get(),
                'date_dest_dir': self.date_dest_dir_entry.get(),
                'last_organizing_timestamp': self.last_organizing_timestamp,
                'organizer_name': self.organizer_name_entry.get()
            }
            with open(PATHS_FILE, 'w') as f:
                json.dump(paths, f, indent=4)
        except Exception as e:
            self.log_text.insert(tk.END, f"Error saving paths: {e}\n")
            self.log_text.see(tk.END)

    def _get_existing_patient_info(self, dest_dir):
        existing_patients = []
        if os.path.isdir(dest_dir):
            for folder_name in os.listdir(dest_dir):
                # Assuming folder names are in "Patient Name Patient ID" format
                match = re.match(r"^(.*)\s+(\S+)$", folder_name)
                if match:
                    name = match.group(1).strip()
                    patient_id = match.group(2).strip()
                    existing_patients.append({"name": name, "id": patient_id, "folder_name": folder_name})
        return existing_patients

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
        source_dir = self.source_dir_entry.get()
        dest_dir = self.dest_dir_entry.get()
        date_dest_dir = self.date_dest_dir_entry.get()
        organizer_name = self.organizer_name_entry.get().strip()

        if not os.path.isdir(source_dir):
            messagebox.showerror("Error", "Source directory does not exist.")
            return
        if not os.path.isdir(dest_dir):
            messagebox.showerror("Error", "Destination directory does not exist.")
            return
        if not os.path.isdir(date_dest_dir):
            messagebox.showerror("Error", "Date Destination directory does not exist.")
            return

        # Determine the start timestamp based on selected mode
        start_timestamp = 0.0
        if self.start_mode_var.get() == "auto":
            start_timestamp = self.last_organizing_timestamp
            self.log_text.insert(tk.END, f"Starting organization from last completed run: {datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n")
        else: # Manual mode
            manual_date_str = self.manual_date_entry.get().strip()
            if not manual_date_str:
                messagebox.showerror("Input Error", "Please enter a date for manual start.")
                return
            try:
                start_datetime = datetime.strptime(manual_date_str, '%Y-%m-%d')
                start_timestamp = start_datetime.timestamp()
                self.log_text.insert(tk.END, f"Starting organization from manual date: {manual_date_str}\n")
            except ValueError:
                messagebox.showerror("Input Error", "Invalid date format. Please use YYYY-MM-DD.")
                return
        self.log_text.see(tk.END)

        self.is_organizing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"Starting organization from {source_dir} to {dest_dir} and {date_dest_dir}\n")
        self.log_text.see(tk.END)
        
        # Run the organizing process in a separate thread to keep GUI responsive
        import threading
        organizing_thread = threading.Thread(target=self._start_organizing_thread, args=(source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp))
        organizing_thread.start()

    def _start_organizing_thread(self, source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp):
        try:
            self.classify_and_organize(source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp)
            self.last_organizing_timestamp = datetime.now().timestamp() # Update timestamp after successful organization
            self.save_paths() # Save paths after organizing
        finally:
            self.is_organizing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.log_text.insert(tk.END, "Organization process finished or stopped.\n")
            self.log_text.see(tk.END)

    def stop_organizing(self):
        self.is_organizing = False
        self.log_text.insert(tk.END, "Stopping organization process...\n")
        self.log_text.see(tk.END)

    def classify_and_organize(self, source_dir, dest_dir, date_dest_dir, organizer_name, start_timestamp):
        all_image_files = []
        for filename in sorted(os.listdir(source_dir)):
            file_path = os.path.join(source_dir, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if os.path.getmtime(file_path) > start_timestamp:
                    all_image_files.append(file_path)
                else:
                    self.log_text.insert(tk.END, f"Skipping {filename} (already organized or older than specified start time).\n")
                    self.log_text.see(tk.END)

        self.log_text.insert(tk.END, f"Classifying {len(all_image_files)} images...\n")
        self.log_text.see(tk.END)

        # Re-check stop flag after counting images, before processing
        if not self.is_organizing:
            self.log_text.insert(tk.END, "Organization stopped by user.\n")
            self.log_text.see(tk.END)
            return

        # Store classification results
        classified_results = {} # {image_path: "info" or "not_info"}

        if self.model is None:
            self.log_text.insert(tk.END, "Deep learning model not loaded. Aborting classification.\n")
            self.log_text.see(tk.END)
            return

        self.model.eval()

        for image_path in all_image_files:
            if not self.is_organizing: # Check stop flag
                self.log_text.insert(tk.END, "Organization stopped by user.\n")
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
                self.log_text.insert(tk.END, f"Classified {os.path.basename(image_path)} as {predicted_class} (Confidence: {confidence:.2f})\n")
                self.log_text.see(tk.END)

            except Exception as e:
                self.log_text.insert(tk.END, f"Error classifying image {os.path.basename(image_path)}: {e}\n")
                self.log_text.see(tk.END)
                classified_results[image_path] = "not_info" # Default to clinical if classification fails

        # Now, organize based on classification results
        current_patient_folder = None
        patient_info_count = 0
        clinical_count = 0

        for i, image_path in enumerate(all_image_files):
            if not self.is_organizing: # Check stop flag
                self.log_text.insert(tk.END, "Organization stopped by user.\n")
                return
            
            # Calculate creation_datetime and new_filename for every image
            try:
                image = PIL.Image.open(image_path)
                exif_data = image._getexif()
                if exif_data and 36867 in exif_data:
                    creation_datetime = datetime.strptime(exif_data[36867], '%Y:%m:%d %H:%M:%S')
                else:
                    creation_datetime = datetime.fromtimestamp(os.path.getmtime(image_path))
            except Exception:
                creation_datetime = datetime.fromtimestamp(os.path.getmtime(image_path))

            formatted_date = creation_datetime.strftime('%Y-%m-%d')
            base_name, ext = os.path.splitext(os.path.basename(image_path))
            new_filename = f"{base_name} {formatted_date}{ext}"

            classification_data = classified_results.get(image_path)
            predicted_class = classification_data["class"]
            confidence = classification_data["confidence"]

            # Override: If not_info and confidence < 0.95, always prompt for review
            if predicted_class == "not_info" and confidence < 0.95:
                action = self.review_low_confidence_not_info_image(image_path, predicted_class, confidence)
                if action == "change_to_info":
                    predicted_class = "info" # Override model's prediction
                    self.log_text.insert(tk.END, f"User changed {os.path.basename(image_path)} to 'info' due to low confidence.\n")
                    self.log_text.see(tk.END)
                else:
                    self.log_text.insert(tk.END, f"User confirmed {os.path.basename(image_path)} as 'not_info' despite low confidence.\n")
                    self.log_text.see(tk.END)

            # If review_not_info_var is checked and the model predicted not_info, prompt for review
            elif self.review_not_info_var.get() and predicted_class == "not_info":
                action = self.review_not_info_image(image_path, predicted_class, confidence)
                if action == "change_to_info":
                    predicted_class = "info" # Override model's prediction
                    self.log_text.insert(tk.END, f"User changed {os.path.basename(image_path)} to 'info'.\n")
                    self.log_text.see(tk.END)
                else:
                    self.log_text.insert(tk.END, f"User kept {os.path.basename(image_path)} as 'not_info'.\n")
                    self.log_text.see(tk.END)

            if predicted_class == "info":
                patient_name, patient_id, classified_as_clinical_by_user = self.get_patient_info(image_path)

                if classified_as_clinical_by_user:
                    # User explicitly classified as clinical, override model's prediction
                    predicted_class = "not_info"
                    self.log_text.insert(tk.END, f"User re-classified {os.path.basename(image_path)} as clinical.\n")
                    self.log_text.see(tk.END)
                else:
                    patient_info_count += 1
                    if patient_name and patient_id:
                        # Logic for dest_dir (환자별 분류 폴더)
                        folder_name_patient_id = f"{patient_name} {patient_id}"
                        patient_folder_dest = os.path.join(dest_dir, folder_name_patient_id)
                        os.makedirs(patient_folder_dest, exist_ok=True)

                        # Logic for date_dest_dir (날짜별 분류 폴더)
                        week_day = get_korean_weekday(creation_datetime.weekday())
                        date_folder_name = creation_datetime.strftime(f'%Y년도')
                        month_folder_name = creation_datetime.strftime(f'%m월')
                        day_folder_name = creation_datetime.strftime(f'%Y.%m.%d ({week_day})')

                        # Construct the full path for date_dest_dir
                        target_date_folder = os.path.join(date_dest_dir, date_folder_name, month_folder_name, day_folder_name, organizer_name)
                        os.makedirs(target_date_folder, exist_ok=True)

                        # Create patient folder inside the date folder
                        target_patient_folder_in_date = os.path.join(target_date_folder, folder_name_patient_id)
                        os.makedirs(target_patient_folder_in_date, exist_ok=True)

                        # Copy the image to both destinations
                        shutil.copy2(image_path, os.path.join(patient_folder_dest, new_filename))
                        self.log_text.insert(tk.END, f"Copied patient info file {os.path.basename(image_path)} to {patient_folder_dest} as {new_filename}\n")
                        self.log_text.see(tk.END)

                        shutil.copy2(image_path, os.path.join(target_patient_folder_in_date, new_filename))
                        self.log_text.insert(tk.END, f"Copied patient info file {os.path.basename(image_path)} to {target_patient_folder_in_date} as {new_filename}\n")
                        self.log_text.see(tk.END)

                        current_patient_folder = target_patient_folder_in_date # Set current patient folder for subsequent clinical files
                    else:
                        self.log_text.insert(tk.END, f"Skipping organization for {os.path.basename(image_path)} due to missing patient info.\n")
                        self.log_text.see(tk.END)
                        current_patient_folder = None # Reset if patient info is not complete

            if predicted_class == "not_info": # This will now also catch files re-classified by user
                clinical_count += 1
                if current_patient_folder:
                    # current_patient_folder is like date_dest_dir/YYYY년도/MM월/YYYY.MM.DD (요일)/Patient Name Patient ID
                    # We need to extract Patient Name Patient ID from it to get the patient_folder_dest
                    path_parts = current_patient_folder.split(os.sep)
                    folder_name_patient_id = path_parts[-1] # e.g., "Patient Name Patient ID"
                    
                    # Copy to the date-based patient folder
                    shutil.copy2(image_path, os.path.join(current_patient_folder, new_filename))
                    self.log_text.insert(tk.END, f"Copied clinical file {os.path.basename(image_path)} to {current_patient_folder} as {new_filename}\n")
                    self.log_text.see(tk.END)

                    # Copy to the patient-only folder in dest_dir
                    patient_folder_dest = os.path.join(dest_dir, folder_name_patient_id)
                    os.makedirs(patient_folder_dest, exist_ok=True)
                    shutil.copy2(image_path, os.path.join(patient_folder_dest, new_filename))
                    self.log_text.insert(tk.END, f"Copied clinical file {os.path.basename(image_path)} to {patient_folder_dest} as {new_filename}\n")
                    self.log_text.see(tk.END)

                else:
                    # Handle clinical files that appear before any patient info file or after a patient info file that failed to provide info
                    # For now, let's copy them to a generic "unclassified_clinical" folder within dest_dir and date_dest_dir
                    unclassified_folder_dest = os.path.join(dest_dir, "unclassified_clinical")
                    os.makedirs(unclassified_folder_dest, exist_ok=True)
                    shutil.copy2(image_path, os.path.join(unclassified_folder_dest, new_filename))
                    self.log_text.insert(tk.END, f"Copied unclassified clinical file {os.path.basename(image_path)} to {unclassified_folder_dest} as {new_filename}\n")
                    self.log_text.see(tk.END)

                    # Also copy to unclassified_clinical in date_dest_dir
                    unclassified_folder_date_dest = os.path.join(date_dest_dir, "unclassified_clinical")
                    os.makedirs(unclassified_folder_date_dest, exist_ok=True)
                    shutil.copy2(image_path, os.path.join(unclassified_folder_date_dest, new_filename))
                    self.log_text.insert(tk.END, f"Copied unclassified clinical file {os.path.basename(image_path)} to {unclassified_folder_date_dest} as {new_filename}\n")
                    self.log_text.see(tk.END)

        self.log_text.insert(tk.END, f"Finished organization. Processed {patient_info_count} patient info files and {clinical_count} clinical files.\n")
        self.log_text.see(tk.END)

    def review_not_info_image(self, image_path, model_prediction, model_confidence):
        review_window = tk.Toplevel(self.root)
        review_window.title("Review 'not_info' Image")
        review_window.geometry("500x600+950+100") # Same position as patient info popup
        review_window.transient(self.root)
        review_window.grab_set()
        review_window.focus_set()

        from PIL import ImageTk
        img = PIL.Image.open(image_path)
        img.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(review_window, image=photo)
        img_label.image = photo
        img_label.pack(pady=10)

        tk.Label(review_window, text=f"Model predicted: {model_prediction} (Confidence: {model_confidence:.2f})").pack()
        tk.Label(review_window, text=f"File: {os.path.basename(image_path)}").pack()

        result = {"action": "keep_not_info"} # Default action

        def on_keep_not_info():
            result["action"] = "keep_not_info"
            review_window.destroy()

        def on_change_to_info():
            result["action"] = "change_to_info"
            review_window.destroy()

        keep_button = tk.Button(review_window, text="Keep as 'not_info' (Enter)", command=on_keep_not_info)
        keep_button.pack(pady=5)
        review_window.bind("<Return>", lambda event=None: on_keep_not_info())

        tk.Button(review_window, text="Change to 'info'", command=on_change_to_info).pack(pady=5)

        review_window.wait_window()
        return result["action"]

    def review_low_confidence_not_info_image(self, image_path, model_prediction, model_confidence):
        review_window = tk.Toplevel(self.root)
        review_window.title("Low Confidence 'not_info' Review")
        review_window.geometry("600x700+950+100") # Same position as patient info popup
        review_window.transient(self.root)
        review_window.grab_set()
        review_window.focus_set()

        from PIL import ImageTk
        img = PIL.Image.open(image_path)
        img.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(review_window, image=photo)
        img_label.image = photo
        img_label.pack(pady=10)

        tk.Label(review_window, text=f"Model predicted: {model_prediction} (Confidence: {model_confidence:.2f})").pack()
        tk.Label(review_window, text="This prediction has low confidence. Please confirm.").pack()
        tk.Label(review_window, text=f"File: {os.path.basename(image_path)}").pack()

        result = {"action": "keep_not_info"} # Default action

        def on_keep_not_info():
            result["action"] = "keep_not_info"
            review_window.destroy()

        def on_change_to_info():
            result["action"] = "change_to_info"
            review_window.destroy()

        keep_button = tk.Button(review_window, text="Confirm as 'not_info' (Enter)", command=on_keep_not_info)
        keep_button.pack(pady=5)
        review_window.bind("<Return>", lambda event=None: on_keep_not_info())

        tk.Button(review_window, text="Change to 'info'", command=on_change_to_info).pack(pady=5)

        review_window.wait_window()
        return result["action"]

    def get_patient_info(self, image_path):
        # OCR 기능을 제거하고 항상 사용자에게 정보를 입력받도록 변경
        self.log_text.insert(tk.END, f"Prompting user for patient info for {os.path.basename(image_path)}\n")
        name, patient_id, classified_as_clinical = self.prompt_for_patient_info(image_path)
        return name, patient_id, classified_as_clinical

    def prompt_for_patient_info(self, image_path):
        # Create a new window for user input
        input_window = tk.Toplevel(self.root)
        input_window.title("Enter Patient Information")
        input_window.geometry("500x620+950+100") # Adjusted height
        input_window.transient(self.root)
        input_window.grab_set()
        input_window.focus_set()

        # Display the image
        from PIL import ImageTk
        img = PIL.Image.open(image_path)
        img.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(input_window, image=photo)
        img_label.image = photo
        img_label.pack(pady=10)

        # Entry fields for name and ID
        tk.Label(input_window, text="Patient Name:").pack()
        name_entry = tk.Entry(input_window, width=40)
        name_entry.pack()
        name_entry.focus_set()
        tk.Label(input_window, text="Patient ID:").pack()
        id_entry = tk.Entry(input_window, width=40)
        id_entry.pack()

        patient_info = {"name": "", "id": "", "classified_as_clinical": False}

        def _reset_id_color_and_unbind(event=None):
            id_entry.config(fg="black")
            id_entry.unbind("<Keypress>") # Unbind after the first keypress

        def _autocomplete_id(event=None):
            name_typed = name_entry.get().strip()
            dest_dir = self.dest_dir_entry.get()
            suggestion_found = None

            if name_typed and os.path.isdir(dest_dir):
                existing_patients = self._get_existing_patient_info(dest_dir)
                for patient in existing_patients:
                    if patient["name"].lower().startswith(name_typed.lower()):
                        suggestion_found = patient["id"]
                        break

            if suggestion_found and not id_entry.get():
                id_entry.delete(0, tk.END)
                id_entry.insert(0, suggestion_found)
                id_entry.config(fg="blue")
                id_entry.bind("<Keypress>", _reset_id_color_and_unbind)

        def on_submit():
            name = name_entry.get().strip()
            patient_id = id_entry.get().strip()

            if not name or not patient_id:
                messagebox.showwarning("Input Error", "Patient Name and Patient ID cannot be empty.")
                return

            patient_info["name"] = name
            patient_info["id"] = patient_id
            input_window.destroy()

        def on_classify_clinical():
            patient_info["classified_as_clinical"] = True
            input_window.destroy()

        submit_button = tk.Button(input_window, text="Submit (Enter in ID field)", command=on_submit)
        submit_button.pack(pady=5)

        classify_clinical_button = tk.Button(input_window, text="임상 파일로 분류", command=on_classify_clinical)
        classify_clinical_button.pack(pady=5)

        def on_stop_organizing_from_popup():
            self.stop_organizing()
            input_window.destroy()

        stop_button_popup = tk.Button(input_window, text="Stop Organizing", command=on_stop_organizing_from_popup)
        stop_button_popup.pack(pady=5)

        # Bind events
        name_entry.bind("<FocusOut>", _autocomplete_id)
        name_entry.bind("<Return>", lambda e: id_entry.focus_set()) # Move focus on Enter
        id_entry.bind("<Return>", lambda e: on_submit())

        input_window.wait_window()

        # After getting initial input, check for similar existing patients
        if not patient_info["classified_as_clinical"] and patient_info["name"] and patient_info["id"]:
            dest_dir = self.dest_dir_entry.get()
            existing_patients = self._get_existing_patient_info(dest_dir)
            
            exact_match_found = any(
                p["name"].lower() == patient_info["name"].lower() and 
                p["id"].lower() == patient_info["id"].lower() 
                for p in existing_patients
            )

            if not exact_match_found:
                similar_suggestions = self._find_similar_patients(patient_info["name"], patient_info["id"], existing_patients)
                if similar_suggestions:
                    chosen_name, chosen_id = self._prompt_for_suggestion(similar_suggestions, patient_info["name"], patient_info["id"])
                    patient_info["name"] = chosen_name
                    patient_info["id"] = chosen_id

        return patient_info["name"], patient_info["id"], patient_info["classified_as_clinical"]

    def _prompt_for_suggestion(self, suggestions, original_name, original_id):
        suggestion_window = tk.Toplevel(self.root)
        suggestion_window.title("Did you mean...?")
        suggestion_window.geometry("400x300+1460+100") # Position next to the patient info popup
        suggestion_window.transient(self.root)
        suggestion_window.grab_set()
        suggestion_window.focus_set()

        tk.Label(suggestion_window, text="Did you mean one of these existing patients?").pack(pady=10)

        selected_option = tk.StringVar(value="original")

        for i, patient in enumerate(suggestions):
            text = f"{patient['name']} {patient['id']}"
            tk.Radiobutton(suggestion_window, text=text, variable=selected_option, value=f"suggestion_{i}").pack(anchor="w")

        tk.Radiobutton(suggestion_window, text=f"No, use my input: {original_name} {original_id}", variable=selected_option, value="original").pack(anchor="w")

        result = {"name": original_name, "id": original_id}

        def on_select():
            choice = selected_option.get()
            if choice.startswith("suggestion_"):
                index = int(choice.split("_")[1])
                result["name"] = suggestions[index]["name"]
                result["id"] = suggestions[index]["id"]
            suggestion_window.destroy()

        tk.Button(suggestion_window, text="Select", command=on_select).pack(pady=10)

        suggestion_window.wait_window()
        return result["name"], result["id"]


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoOrganizerApp(root)
    root.mainloop()
