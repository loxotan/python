# 수술 영상 머리삭제기 v1.9 (최종 포터블)
# CUDA 지원 여부를 자동 감지하여 NVENC(GPU) 또는 libx264(CPU) 인코더로 자동 전환합니다.

import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms as T
import threading
import queue
import numpy as np
import subprocess

# --- GUI 라이브러리 ---
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# --- 설정 변수 ---
HEAD_REMOVER_MODEL_PATH = "head_remover_model.pth"
MOUTH_DETECTOR_MODEL_PATH = "mouth_detector_model.pth"
CONFIDENCE_THRESHOLD = 0.7

# --- 1단계 설정 ---
TARGET_FPS_HEAD = 2

# --- 2단계 설정 ---
TARGET_FPS_MOUTH = 0.05 # 20초에 1번
OUTPUT_W, OUTPUT_H = 1920, 1080
ROI_TARGET_OCCUPANCY = 0.75
CAM_SMOOTHING = 0.08

# --- 모델 불러오기 함수들 ---
def load_head_remover_model(model_path, device):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.classifier[1].in_features, 1))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_mouth_detector_model(model_path, device):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- 영상 처리 로직 ---
def predict_frame_to_keep(model, device, frame):
    transform = T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(frame_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        return torch.sigmoid(output).item() > 0.5

def pass1_head_removal_ffmpeg(input_path, temp_output_path, model, device, progress_queue, codec, preset):
    progress_queue.put(("status", "1/2단계: 머리 제거 구간 분석 중 (여유시간 적용)..."))
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise FileNotFoundError(f"입력 파일 '{input_path}'를 열 수 없습니다.")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    frame_skip_interval = max(1, int(original_fps / TARGET_FPS_HEAD))

    PRE_PADDING, POST_PADDING = 0.5, 1.0
    good_segments = []
    is_good, segment_start_time = True, 0

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        if frame_count % frame_skip_interval == 0:
            current_time = frame_count / original_fps
            should_keep = predict_frame_to_keep(model, device, frame)
            if is_good and not should_keep:
                end_time = max(segment_start_time, current_time - PRE_PADDING)
                if end_time > segment_start_time: good_segments.append((segment_start_time, end_time))
                is_good = False
            elif not is_good and should_keep:
                segment_start_time = current_time + POST_PADDING
                is_good = True
        if frame_count % 30 == 0: progress_queue.put(("status", f"1/2단계: 분석 중... {frame_count}/{total_frames}"))

    if is_good and duration > segment_start_time: good_segments.append((segment_start_time, duration))
    cap.release()

    if not good_segments: raise ValueError("머리 제거 후 남는 영상 구간이 없습니다.")

    progress_queue.put(("status", f"1/2단계: FFmpeg ({codec}) 가속으로 임시 파일 생성 중..."))
    
    filter_str = ""
    concat_inputs = ""
    for i, (start, end) in enumerate(good_segments):
        filter_str += f"[0:v]trim={start}:{end},setpts=PTS-STARTPTS[v{i}];[0:a]atrim={start}:{end},asetpts=PTS-STARTPTS[a{i}];"
        concat_inputs += f"[v{i}][a{i}]"
    filter_str += f"{concat_inputs}concat=n={len(good_segments)}:v=1:a=1[outv][outa]"

    command = ['ffmpeg', '-i', input_path, '-filter_complex', filter_str, '-map', '[outv]', '-map', '[outa]', '-c:v', codec] + preset + ['-y', temp_output_path]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    progress_queue.put(("progress", 50))
    return True

def pass2_mouth_tracking_opencv(temp_input_path, final_output_path, model, device, progress_queue, codec, preset):
    progress_queue.put(("status", f"2/2단계: 입 추적 및 확대 중 ({codec} 가속)..."))
    cap = cv2.VideoCapture(temp_input_path)
    if not cap.isOpened(): raise FileNotFoundError(f"임시 파일 '{temp_input_path}'를 열 수 없습니다.")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip_interval = max(1, int(original_fps / TARGET_FPS_MOUTH))

    command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OUTPUT_W}x{OUTPUT_H}', '-r', str(original_fps), '-i', '-', '-c:v', codec] + preset + ['-pix_fmt', 'yuv420p', final_output_path]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ret, first_frame = cap.read()
    if not ret: raise ValueError("임시 파일을 읽을 수 없습니다.")
    transform_mouth = T.Compose([T.ToTensor()])
    img_tensor = transform_mouth(first_frame).to(device)
    with torch.no_grad(): prediction = model([img_tensor])
    best_box = None; max_score = 0
    for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
        if score > CONFIDENCE_THRESHOLD and score > max_score: best_box, max_score = box, score
    fixed_crop_w = (best_box[2] - best_box[0]) / ROI_TARGET_OCCUPANCY if best_box is not None else OUTPUT_W / 1.5
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    cam_x, cam_y = orig_w / 2, orig_h / 2
    target_x, target_y = orig_w / 2, orig_h / 2

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        if frame_count % frame_skip_interval == 0:
            img_tensor = transform_mouth(frame).to(device)
            with torch.no_grad(): prediction = model([img_tensor])
            best_box = None; max_score = 0
            for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
                if score > CONFIDENCE_THRESHOLD and score > max_score: best_box, max_score = box, score
            if best_box is not None: target_x, target_y = (best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2

        cam_x += (target_x - cam_x) * CAM_SMOOTHING
        cam_y += (target_y - cam_y) * CAM_SMOOTHING
        current_crop_w = fixed_crop_w
        current_crop_h = current_crop_w * (OUTPUT_H / OUTPUT_W)

        crop_x1 = int(cam_x - current_crop_w / 2)
        crop_y1 = int(cam_y - current_crop_h / 2)
        crop_x1 = max(0, min(crop_x1, orig_w - int(current_crop_w)))
        crop_y1 = max(0, min(crop_y1, orig_h - int(current_crop_h)))
        crop_x2, crop_y2 = crop_x1 + int(current_crop_w), crop_y1 + int(current_crop_h)

        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2: continue

        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        resized_frame = cv2.resize(cropped_frame, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LANCZOS4)
        proc.stdin.write(resized_frame.tobytes())

        if frame_count % 20 == 0:
            progress = 50 + (frame_count / total_frames) * 50
            progress_queue.put(("progress", progress))
            progress_queue.put(("status", f"2/2단계: 입 추적 중... {frame_count}/{total_frames}"))

    proc.stdin.close(); proc.wait(); cap.release()
    return True

# --- GUI 클래스 ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("수술 영상 처리기 v1.9 (최종 포터블)")
        self.geometry("500x250")
        self.input_path, self.output_path = tk.StringVar(), tk.StringVar()
        self.progress_queue = queue.Queue()
        self.create_widgets()
        self.process_queue_updates()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main_frame, text="입력 동영상 (원본 파일):").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(main_frame, textvariable=self.input_path, width=50).grid(row=1, column=0, sticky="ew")
        ttk.Button(main_frame, text="파일 선택", command=self.select_input).grid(row=1, column=1, padx=5)
        ttk.Label(main_frame, text="저장할 동영상:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(main_frame, textvariable=self.output_path, width=50).grid(row=3, column=0, sticky="ew")
        ttk.Button(main_frame, text="파일 선택", command=self.select_output).grid(row=3, column=1, padx=5)
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
        self.status_label = ttk.Label(main_frame, text="대기 중...")
        self.status_label.grid(row=5, column=0, columnspan=2, sticky="w")
        self.start_button = ttk.Button(main_frame, text="처리 시작", command=self.start_processing)
        self.start_button.grid(row=6, column=0, columnspan=2, pady=10)
        main_frame.columnconfigure(0, weight=1)

    def select_input(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if path: self.input_path.set(path)

    def select_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if path: self.output_path.set(path)

    def start_processing(self):
        if not self.input_path.get() or not self.output_path.get():
            messagebox.showerror("오류", "입력 및 저장 경로를 모두 선택해주세요.")
            return
        self.start_button.config(state="disabled")
        self.progress["value"] = 0
        thread = threading.Thread(target=self.run_in_thread, args=(self.input_path.get(), self.output_path.get()), daemon=True)
        thread.start()

    def run_in_thread(self, input_p, output_p):
        temp_p = os.path.splitext(input_p)[0] + ".temp.mp4"
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.progress_queue.put(("status", f"사용 장치: {device}"))

            if str(device) == 'cuda':
                ffmpeg_codec, ffmpeg_preset = 'h264_nvenc', ['-preset', 'fast']
            else:
                ffmpeg_codec, ffmpeg_preset = 'libx264', ['-preset', 'ultrafast']
            
            head_model = load_head_remover_model(HEAD_REMOVER_MODEL_PATH, device)
            if not head_model: return
            pass1_success = pass1_head_removal_ffmpeg(input_p, temp_p, head_model, device, self.progress_queue, ffmpeg_codec, ffmpeg_preset)
            
            if pass1_success:
                mouth_model = load_mouth_detector_model(MOUTH_DETECTOR_MODEL_PATH, device)
                if not mouth_model: return
                pass2_success = pass2_mouth_tracking_opencv(temp_p, output_p, mouth_model, device, self.progress_queue, ffmpeg_codec, ffmpeg_preset)
                if pass2_success: self.progress_queue.put(("done", "모든 작업 완료!"))

        except Exception as e:
            self.progress_queue.put(("error", f"처리 중 오류 발생: {e}"))
        finally:
            if os.path.exists(temp_p):
                os.remove(temp_p)

    def process_queue_updates(self):
        try:
            message_type, value = self.progress_queue.get_nowait()
            if message_type == "progress": self.progress["value"] = value
            elif message_type == "status": self.status_label.config(text=value)
            elif message_type == "done":
                self.status_label.config(text=value)
                messagebox.showinfo("완료", value)
                self.start_button.config(state="normal")
            elif message_type == "error":
                self.status_label.config(text=f"오류: {value}")
                messagebox.showerror("오류", value)
                self.start_button.config(state="normal")
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue_updates)

if __name__ == '__main__':
    app = App()
    app.mainloop()
