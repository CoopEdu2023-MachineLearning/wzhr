import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# 设置路径
DATA_DIR = '/Users/wangzihaoran/machineLearning/unit1/pokemon/data'
POKEMON_DIR = os.path.join(DATA_DIR, 'pokemon')
DIGIMON_DIR = os.path.join(DATA_DIR, 'digimon')
MODEL_PATH = '/Users/wangzihaoran/machineLearning/unit1/pokemon/pokemon_digimon_model.h5'

# 图像参数
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32

# 数据增强和准备
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 训练集
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# 验证集
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 构建CNN模型
model = Sequential([
    # 第一个卷积块
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    
    # 第二个卷积块
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # 第三个卷积块
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # 展平层
    Flatten(),
    
    # 全连接层
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 二分类问题使用sigmoid
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 模型摘要
model.summary()

# 回调函数
callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(monitor='val_loss', patience=5)
]

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=30,
    callbacks=callbacks
)

# 绘制训练过程
plt.figure(figsize=(12, 4))

# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('模型准确率')
plt.ylabel('准确率')
plt.xlabel('Epoch')
plt.legend(['训练集', '验证集'], loc='lower right')

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('模型损失')
plt.ylabel('损失')
plt.xlabel('Epoch')
plt.legend(['训练集', '验证集'], loc='upper right')

plt.tight_layout()
plt.savefig('/Users/wangzihaoran/machineLearning/unit1/pokemon/training_history.png')
plt.show()

# 加载最佳模型进行评估
best_model = tf.keras.models.load_model(MODEL_PATH)
test_loss, test_acc = best_model.evaluate(validation_generator)
print(f'测试准确率: {test_acc:.4f}')

# 创建一个函数用于预测新图像
def predict_image(img_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = best_model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Digimon", prediction[0][0]
    else:
        return "Pokemon", 1 - prediction[0][0]

# 示例：如何使用预测函数
# 假设有一张测试图像
# result, confidence = predict_image('/path/to/test/image.jpg')
# print(f'预测结果: {result}，置信度: {confidence:.4f}')