import torch 
# 1. CUDA 사용 가능 여부 확인
is_available = torch.cuda.is_available()
print(f"PyTorch에서 CUDA를 사용할 수 있나요? {is_available}")

if is_available:
    # 2. 사용 가능한 GPU 개수 확인
    gpu_count = torch.cuda.device_count()
    print(f"사용 가능한 GPU 개수: {gpu_count}")
    # 3. 현재 GPU 장치 이름 확인
    gpu_name = torch.cuda.get_device_name(0)
    print(f"현재 GPU 이름: {gpu_name}")
else:
    print("PyTorch가 GPU를 찾지 못했습니다. CPU 버전으로 설치되었을 수 있습니다.")