from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "C:\\Users\\Acer\\OneDrive\\Masaüstü\\ödev3\\images\\daire.png"

image = Image.open(image_path).convert('L')  # Convert to grayscale for processing

# Convert image to numpy array
image_array = np.array(image)

# Define the horizontal derivative filter [-1, 1]
filter_hor = np.array([-1, 1])

# Apply the filter to the image (horizontal derivative)
filtered_image = np.zeros_like(image_array)

for i in range(image_array.shape[0]):
    for j in range(1, image_array.shape[1]):
        filtered_image[i, j] = image_array[i, j] - image_array[i, j - 1]

# Display the original and filtered image
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Original Image
axs[0].imshow(image_array, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

# Filtered Image (Horizontal Derivative)
axs[1].imshow(filtered_image, cmap='gray')
axs[1].set_title("Filtered Image (Horizontal Derivative)")
axs[1].axis('off')

plt.show()
