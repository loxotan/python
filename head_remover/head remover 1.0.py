# 수술 영상 머리삭제기 v1.3 (PyTorch + OpenCV GUI)
# MoviePy 의존성을 제거하고 OpenCV만으로 동영상을 처리합니다.

import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import threading
import queue

# --- GUI 라이브러리 ---
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# --- 설정 변수 ---
MODEL_PATH = "c:\\python\\head_remover_model.pth"
IMG_WIDTH, IMG_HEIGHT = 224, 224
TARGET_FPS = 2 # 분석 정밀도. 높일수록 정확하지만 느려짐

# --- 모델 로딩 (이전과 동일) ---
def load_model(model_path, progress_queue):
    try:
        progress_queue.put(("status", "모델 불러오는 중..."))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        progress_queue.put(("status", f"모델 로드 완료. ({device})"))
        return model, device
    except FileNotFoundError:
        progress_queue.put(("error", f"모델 파일 '{model_path}'를 찾을 수 없습니다."))
        return None, None
    except Exception as e:
        progress_queue.put(("error", f"모델 로드 실패: {e}"))
        return None, None

# --- 영상 처리 로직 (OpenCV) ---
def predict_frame(model, device, frame):
    transform = transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.CenterCrop((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        return torch.sigmoid(output).item() > 0.5

def process_video_opencv(input_path, output_path, model, device, progress_queue):
    try:
        progress_queue.put(("status", "동영상 처리 준비 중..."))
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            progress_queue.put(("error", f"입력 파일 '{input_path}'를 열 수 없습니다."))
            return

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip_interval = max(1, int(original_fps / TARGET_FPS))

        # 출력 동영상의 FPS를 TARGET_FPS로 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (frame_width, frame_height))

        frame_count = 0
        saved_frame_count = 0

        for current_frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # 분석 주기마다 프레임 처리
            if current_frame_num % frame_skip_interval == 0:
                if predict_frame(model, device, frame):
                    out.write(frame)
                    saved_frame_count += 1
            
            if current_frame_num % 10 == 0:
                progress_queue.put(("progress", (current_frame_num / total_frames) * 100))
                progress_queue.put(("status", f"처리 중... {current_frame_num}/{total_frames} (저장: {saved_frame_count})"))

        cap.release()
        out.release()
        progress_queue.put(("progress", 100))
        progress_queue.put(("done", f"처리 완료! 총 {total_frames} 프레임 중 {saved_frame_count}개를 저장했습니다."))

    except Exception as e:
        progress_queue.put(("error", f"처리 중 오류 발생: {e}"))

# --- GUI 클래스 (이전과 거의 동일) ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("수술 영상 머리삭제기 v1.0")
        self.geometry("500x250")
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.progress_queue = queue.Queue()
        self.create_widgets()
        self.process_queue_updates()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main_frame, text="입력 동영상:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(main_frame, textvariable=self.input_path, width=50).grid(row=1, column=0, sticky="ew")
        ttk.Button(main_frame, text="파일 선택", command=self.select_input).grid(row=1, column=1, padx=5)
        ttk.Label(main_frame, text="저장할 동영상:").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(main_frame, textvariable=self.output_path, width=50).grid(row=3, column=0, sticky="ew")
        ttk.Button(main_frame, text="파일 선택", command=self.select_output).grid(row=3, column=1, padx=5)
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
        self.status_label = ttk.Label(main_frame, text="대기 중...")
        self.status_label.grid(row=5, column=0, columnspan=2, sticky="w")
        self.start_button = ttk.Button(main_frame, text="처리 시작", command=self.start_processing)
        self.start_button.grid(row=6, column=0, columnspan=2, pady=10)
        main_frame.columnconfigure(0, weight=1)

    def select_input(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
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
        model, device = load_model(MODEL_PATH, self.progress_queue)
        if model and device:
            process_video_opencv(input_p, output_p, model, device, self.progress_queue)

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