import os
import random
import matplotlib.pyplot as plt
from pokemon_digimon_classifier import predict_image

# 设置测试图像路径
DATA_DIR = '/Users/wangzihaoran/machineLearning/unit1/pokemon/data'
POKEMON_DIR = os.path.join(DATA_DIR, 'pokemon')
DIGIMON_DIR = os.path.join(DATA_DIR, 'digimon')

# 随机选择一些图像进行测试
def get_random_images(directory, n=5):
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(root, file))
    return random.sample(images, min(n, len(images)))

# 获取随机图像
pokemon_images = get_random_images(POKEMON_DIR)
digimon_images = get_random_images(DIGIMON_DIR)
test_images = pokemon_images + digimon_images
random.shuffle(test_images)

# 显示预测结果
plt.figure(figsize=(15, 10))
for i, img_path in enumerate(test_images):
    # 预测
    result, confidence = predict_image(img_path)
    
    # 显示图像和预测结果
    img = plt.imread(img_path)
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    plt.title(f"预测: {result}\n置信度: {confidence:.2f}")
    plt.axis('off')
    
    # 真实标签
    true_label = "Pokemon" if "pokemon" in img_path.lower() else "Digimon"
    color = "green" if result == true_label else "red"
    plt.xlabel(f"实际: {true_label}", color=color)

plt.tight_layout()
plt.savefig('/Users/wangzihaoran/machineLearning/unit1/pokemon/prediction_results.png')
plt.show()

# 计算准确率
correct = 0
for img_path in test_images:
    result, _ = predict_image(img_path)
    true_label = "Pokemon" if "pokemon" in img_path.lower() else "Digimon"
    if result == true_label:
        correct += 1

accuracy = correct / len(test_images)
print(f"测试准确率: {accuracy:.2f}")