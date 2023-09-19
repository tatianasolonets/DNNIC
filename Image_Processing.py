import os
from PIL import Image
import numpy as np

# Define a function to read and preprocess the images
def preprocess_image(image_path, num_px):
    # Resize the image to a fixed size.
    img = np.array(Image.open(image_path).resize((num_px, num_px)))
    # Normalize the pixel values to be in the range [0, 1]
    img = img / 255.0
    img = img.reshape((1, num_px * num_px * 3)).T

    return img

# Loop through the files in the train folder
def preprocess_images(folder_path, label_name_1, label_name_0, num_px):
    set_x = []
    set_y = []
    for filename in os.listdir(folder_path):
        if label_name_1 in filename:
            label = 1  # 1 represents a cat
        elif label_name_0 in filename:
            label = 0  # 0 represents a dog
        else:
            continue  # Skip files that don't contain 1 or 0 labels in the name

        # Get the full path to the image file
        image_path = os.path.join(folder_path, filename)

        # Preprocess the image
        img = preprocess_image(image_path, num_px)

        # Append the image and label to the lists
        set_x.append(img)
        set_y.append(label)

    # Convert the lists to NumPy arrays
    set_x = np.array(set_x)
    set_y = np.array(set_y)

    set_x_flatten = set_x.reshape(set_x.shape[0],-1).T
    set_y_flatten = set_y.reshape(set_y.shape[0],-1).T

    print("set_x_flatten data shape from" + str(folder_path) + ": ", set_x_flatten.shape)
    print("set_y_flatten data shape from" + str(folder_path) + ": ", set_y_flatten.shape)

    return set_x_flatten, set_y_flatten
