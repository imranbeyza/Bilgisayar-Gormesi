from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Load necessary libraries
image_path = "C:\\Users\\Acer\\OneDrive\\Masaüstü\\ödev3\\images\\kare.png"

# Open the image and convert to grayscale
image = Image.open(image_path).convert('L')

# Convert image to numpy array
image_array = np.array(image)

# Define the vertical derivative filter [-1, 1]^T
filter_vert = np.array([[-1], [1]])

# Apply the filter to the image (vertical derivative)
filtered_image_vertical = np.zeros_like(image_array)

for i in range(1, image_array.shape[0]):
    for j in range(image_array.shape[1]):
        filtered_image_vertical[i, j] = image_array[i, j] - image_array[i - 1, j]

# Display the original and filtered image
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Original Image
axs[0].imshow(image_array, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis('off')

# Filtered Image (Vertical Derivative)
axs[1].imshow(filtered_image_vertical, cmap='gray')
axs[1].set_title("Filtered Image (Vertical Derivative)")
axs[1].axis('off')

plt.show()
