import torch

# .pth 파일 경로 설정 (여기에 사용하려는 파일의 경로를 넣으세요)
file_path = 'c:/Users/user/Desktop/number_detection_model.pth'

try:
    # .pth 파일 로드
    model_weights = torch.load(file_path)

    # 로드된 데이터 타입 확인
    print(f"Loaded model type: {type(model_weights)}")
    
    # state_dict 형태일 경우, 가중치 정보 출력
    if isinstance(model_weights, dict):
        print("\nModel contains the following layers and their weights:\n")
        for key, value in model_weights.items():
            print(f"{key}: {value.shape}")
    else:
        print("\nThe loaded file is not a state_dict or weight dictionary. It might be a complete model or other data.")

except Exception as e:
    print(f"Error loading the .pth file: {e}")

# 특정 레이어의 가중치를 보고 싶은 경우 (키 이름을 확인 후 사용하세요)
# layer_name = "layer1.weight"  # 실제 레이어 이름으로 수정하세요
# weights = model_weights.get(layer_name)
# if weights is not None:
#     print(f"\nWeights for layer '{layer_name}':\n{weights}")
# else:
#     print(f"Layer '{layer_name}' not found in model.")
