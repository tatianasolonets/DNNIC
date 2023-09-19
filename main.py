from Model import *
from Image_Processing import *

# Set the path to the folder containing your images
train_folder_path = "./data/train/"
dev_folder_path = "./data/dev/"

# Initialize empty lists to store images and labels
train_set_x = []
train_set_y = []
dev_set_x = []
dev_set_y = []

# Loop through the files in the train folder
train_set_x, train_set_y = preprocess_images(train_folder_path, "cat", "dog", 64)
# Loop through the files in the dev folder
dev_set_x, dev_set_y = preprocess_images(dev_folder_path, "cat", "dog", 64)

logistic_regression_model = logistic_regression_model(train_set_x, train_set_y, dev_set_x, dev_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
