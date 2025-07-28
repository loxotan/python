import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# --- 1. 설정값 (Configuration) ---
# 데이터셋이 있는 상위 폴더 경로를 지정하세요.
DATA_DIR = 'dataset' 

# 모델 학습을 위한 파라미터
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15 # 초기 학습 횟수, 필요시 조절

# --- 2. 데이터셋 로드 및 준비 ---
def load_datasets():
    """데이터셋 폴더에서 훈련, 검증, 테스트 데이터를 로드합니다."""
    train_dir = os.path.join(DATA_DIR, 'train')
    validation_dir = os.path.join(DATA_DIR, 'validation')
    test_dir = os.path.join(DATA_DIR, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(validation_dir) or not os.path.exists(test_dir):
        print(f"Error: Make sure the dataset directory structure is correct in '{DATA_DIR}'")
        print("Expected structure: dataset/{train, validation, test}/{patient_info, clinical}")
        return None, None, None

    # Keras 유틸리티를 사용하여 폴더에서 바로 데이터를 로드
    train_dataset = image_dataset_from_directory(
        train_dir,
        label_mode='binary',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = image_dataset_from_directory(
        validation_dir,
        label_mode='binary',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    test_dataset = image_dataset_from_directory(
        test_dir,
        label_mode='binary',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False # 테스트 시에는 섞지 않음
    )
    
    # 클래스 이름 확인 (patient_info, clinical)
    class_names = train_dataset.class_names
    print(f"Class names: {class_names}")

    # 데이터 로딩 성능 향상을 위한 prefetch 설정
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names

# --- 3. 모델 구축 (전이 학습) ---
def build_model(input_shape):
    """MobileNetV2를 기반으로 한 전이 학습 모델을 구축합니다."""
    # 기반 모델 로드 (ImageNet으로 사전 학습된 가중치 사용)
    base_model = MobileNetV2(input_shape=input_shape,
                             include_top=False, # 상위 분류층은 제외
                             weights='imagenet')

    # 기반 모델의 가중치는 고정 (학습되지 않도록)
    base_model.trainable = False

    # 새로운 분류층 추가
    inputs = tf.keras.Input(shape=input_shape)
    # MobileNetV2에 맞는 전처리 레이어
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x) # 과적합 방지를 위한 드롭아웃
    # 최종 출력 레이어 (이진 분류이므로 sigmoid 사용)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    
    return model

# --- 4. 학습 과정 시각화 ---
def plot_history(history):
    """모델 학습 과정의 정확도와 손실을 그래프로 출력합니다."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# --- 5. 메인 실행 로직 ---
def main():
    # 데이터셋 로드
    train_ds, val_ds, test_ds, class_names = load_datasets()
    if train_ds is None:
        return

    # 모델 구축
    model = build_model(IMAGE_SIZE + (3,))
    
    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # 모델 학습
    print("\n--- Start Training ---")
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        validation_data=val_ds)
    print("\n--- Training Finished ---")

    # 학습 결과 시각화
    plot_history(history)

    # 모델 평가 (테스트 데이터 사용)
    print("\n--- Evaluating Model ---")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # 학습된 모델 저장
    model.save('photo_classifier_model.h5')
    print("\nModel saved to photo_classifier_model.h5")
    print("You can now use this model in your GUI application.")


if __name__ == '__main__':
    main()
