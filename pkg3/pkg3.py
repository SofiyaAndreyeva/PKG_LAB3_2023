import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = "/Users/sonyands/Desktop/bsu/лаб пкг/pkg3/6273737733.jpg"


image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Операция: Добавление целочисленной константы к изображению
constant_value = 50
added_image = cv2.add(image, constant_value)

# Операция: Преобразование изображения в негатив
negative_image = 255 - image

# Операция: Умножение изображения на константу
constant_multiplier = 2.0
multiplied_image = cv2.multiply(image, constant_multiplier)

# Операция: Степенное преобразование
power = 0.5  
power_transformed_image = np.power(image, power)

# Операция: Логарифмическое преобразование
log_transformed_image = np.log1p(image)

# Построение гистограммы изображения
hist, bins = np.histogram(image, bins=256, range=(0, 256))

# Эквализация гистограммы
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Линейное контрастирование
alpha = 1.5  
beta = 20   
contrast_image = alpha * image + beta

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

images = [image, added_image, negative_image, multiplied_image, power_transformed_image, log_transformed_image, hist, cdf_normalized, contrast_image]
titles = ['Исходное изображение', 'Добавление константы', 'Негатив', 'Умножение на константу', 'Степенное преобразование', 'Лог. преобразование', 'Гистограмма', 'Экв. гистограмма', 'Линейное контрастирование']

for i in range(9):  # Now we have 9 plots
    if i == 6:
        axes[i].bar(range(256), hist, color='b', alpha=0.5)
    elif i == 7:
        axes[i].plot(cdf_normalized, color='b')
        axes[i].hist(image.ravel(), bins=256, range=(0, 256), color='r', alpha=0.5)
        axes[i].legend(('CDF', 'Гистограмма'), loc='upper left')
    else:
        axes[i].imshow(images[i], cmap='gray')
    axes[i].set_title(titles[i])


fig.delaxes(axes[9])

plt.subplots_adjust(wspace=0.5, hspace=0.5)  
plt.show()