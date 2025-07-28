import torch

if torch.cuda.is_available():
    print(f'현재 디바이스 : {torch.cuda.current_device()}')
    print(f'디바이스 이름 : {torch.cuda.get_device_name()}')
    print(f'디바이스 기능 : {torch.cuda.get_device_capability()}')
    print(f'현재 쿠다 버전: {torch.version.cuda}')
    print(f'현재 파이토치 버전: {torch.__version__}')
else:
    print("No Gpu!")