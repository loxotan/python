# head remover crop model maker.py (v1.2)
# TypeError를 해결하기 위해 Dataset과 Transform 로직을 수정합니다.

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm

# --- 설정 변수 ---
DATA_PATH = "c:\\python\\mouth_data"
MODEL_SAVE_PATH = "mouth_detector_model.pth"
VALIDATION_IMAGE_SAVE_PATH = "validation_outputs"
NUM_CLASSES = 2  # 1 (mouth) + 1 (background)
NUM_EPOCHS = 10
BATCH_SIZE = 4

# --- 1. Custom Dataset 클래스 (수정됨) ---
class MouthDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.file_stems = []
        all_files = os.listdir(root)
        stems = sorted(list(set([os.path.splitext(file)[0] for file in all_files])))
        for stem in stems:
            img_path_jpg = os.path.join(root, stem + '.jpg')
            img_path_png = os.path.join(root, stem + '.png')
            annot_path = os.path.join(root, stem + '.xml')
            if (os.path.exists(img_path_jpg) or os.path.exists(img_path_png)) and os.path.exists(annot_path):
                self.file_stems.append(stem)

    def __getitem__(self, idx):
        stem = self.file_stems[idx]
        img_path_jpg = os.path.join(self.root, stem + '.jpg')
        img_path_png = os.path.join(self.root, stem + '.png')
        img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
        annot_path = os.path.join(self.root, stem + '.xml')
        
        img = Image.open(img_path).convert("RGB")
        tree = ET.parse(annot_path)
        root = tree.getroot()
        boxes = []
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            xmin, ymin, xmax, ymax = [int(bndbox.find(coord).text) for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        # Transform은 이미지에만 적용
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.file_stems)

# --- 2. 모델 및 데이터로더 준비 (수정됨) ---
def get_transform(train):
    # 표준적인 torchvision transform을 사용합니다.
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

# --- 3. 메인 실행 함수 (이전과 동일) ---
def main():
    print("--- 입(mouth) 탐지 모델 훈련 시작 ---")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"사용 장치: {device}")

    os.makedirs(VALIDATION_IMAGE_SAVE_PATH, exist_ok=True)

    dataset = MouthDataset(root=DATA_PATH, transforms=get_transform(train=True))
    dataset_val = MouthDataset(root=DATA_PATH, transforms=get_transform(train=False)) # Validation은 transform을 공유해도 무방
    
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(len(dataset) * 0.8)
    dataset_train = torch.utils.data.Subset(dataset, indices[:train_size])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[train_size:])

    data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"훈련 데이터: {len(dataset_train)}개, 검증 데이터: {len(dataset_val)}개")

    model = get_model(NUM_CLASSES)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(data_loader_train, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [훈련]")
        for images, targets in progress_bar:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
        
        lr_scheduler.step()
        print(f"Epoch #{epoch+1} 평균 훈련 손실: {epoch_loss/len(data_loader_train):.4f}")

        model.eval()
        # 검증셋의 첫번째 이미지를 가져오기 위해 Subset의 내부 dataset과 index에 접근
        val_subset_idx = dataset_val.indices[0]
        val_image, _ = dataset_val.dataset[val_subset_idx]

        with torch.no_grad():
            prediction = model([val_image.to(device)])
        
        img_np = val_image.mul(255).permute(1, 2, 0).byte().cpu().numpy()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
            if score > 0.7:
                x1, y1, x2, y2 = [int(c) for c in box]
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        save_name = os.path.join(VALIDATION_IMAGE_SAVE_PATH, f"epoch_{epoch+1}_prediction.png")
        cv2.imwrite(save_name, img_np)
        print(f"검증 이미지가 '{save_name}'에 저장되었습니다.")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"--- 훈련 완료 ---")
    print(f"모델이 '{MODEL_SAVE_PATH}' 파일로 저장되었습니다.")

if __name__ == '__main__':
    main()
