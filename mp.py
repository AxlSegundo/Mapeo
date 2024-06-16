import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imágenes
color_image = cv2.imread('IMG/cityscape.jpg')
gray_image = cv2.imread('IMG/paisaje.jpg', cv2.IMREAD_GRAYSCALE)

# Mostrar imágenes usando matplotlib para mejor visualización en notebooks
def show_image(img, title=''):
    if len(img.shape) == 3:  # Imagen a color
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
    else:  # Imagen en escala de grises
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Transformación cilíndrica
def cylindrical_transform(image):
    rows, cols, _ = image.shape
    transformed_image = np.zeros_like(image)

    # Transformación en X para cuadrantes I y IV
    for i in range(rows):
        for j in range(cols // 2):
            x = int(j + 30 * np.sin(2 * np.pi * i / 100))
            if x < cols // 2:
                transformed_image[i, j] = image[i, x]
            else:
                transformed_image[i, j] = 0

    # Transformación en Y para cuadrantes II y III
    for i in range(rows):
        for j in range(cols // 2, cols):
            y = int(i + 30 * np.sin(2 * np.pi * j / 100))
            if y < rows:
                transformed_image[i, j] = image[y, j]
            else:
                transformed_image[i, j] = 0

    return transformed_image

# Transformación de ondas
def wave_transform(image, amplitude=30, frequency=20):
    rows, cols, _ = image.shape
    transformed_image = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(amplitude * np.sin(2 * np.pi * i / frequency))
            offset_y = int(amplitude * np.cos(2 * np.pi * j / frequency))
            new_x = (j + offset_x) % cols
            new_y = (i + offset_y) % rows
            transformed_image[new_y, new_x] = image[i, j]

    return transformed_image

# Transformación de apretar o pinchar
def pinch_transform(image, strength=0.5, radius=200):
    rows, cols, _ = image.shape
    transformed_image = np.zeros_like(image)

    center_x, center_y = cols // 2, rows // 2

    for i in range(rows):
        for j in range(cols):
            dx = j - center_x
            dy = i - center_y
            distance = np.sqrt(dx*dx + dy*dy)

            if distance < radius:
                factor = 1 - strength * (radius - distance) / radius
                new_x = int(center_x + dx * factor)
                new_y = int(center_y + dy * factor)

                if 0 <= new_x < cols and 0 <= new_y < rows:
                    transformed_image[i, j] = image[new_y, new_x]
                else:
                    transformed_image[i, j] = 0
            else:
                transformed_image[i, j] = image[i, j]

    return transformed_image

# Utilización de imagen en escala de grises como superficie deformante
def map_transform(image, gray_image):
    rows, cols = gray_image.shape
    transformed_image = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            offset = gray_image[i, j] // 5
            new_x = (j + offset) % cols
            new_y = (i + offset) % rows
            transformed_image[new_y, new_x] = image[i, j]

    return transformed_image

# Aplicar transformaciones
transformed_cylindrical = cylindrical_transform(color_image)
transformed_wave = wave_transform(color_image)
transformed_pinch = pinch_transform(color_image)
transformed_map = map_transform(color_image, gray_image)

# Mostrar resultados
show_image(transformed_cylindrical, 'Cylindrical Transform')
show_image(transformed_wave, 'Wave Transform')
show_image(transformed_pinch, 'Pinch Transform')
show_image(transformed_map, 'Mapped Transform')
