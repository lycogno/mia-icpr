import pandas as pd
import os
import shutil

# Function to create folders if they don't exist
def create_folders(output_folder, num_classes):
    for i in range(num_classes):
        folder_path = os.path.join(output_folder, f"{i}")
        os.makedirs(folder_path, exist_ok=True)

# Function to sort images based on the CSV file
def sort_images(csv_path, input_folder, output_folder):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Create output folders if they don't exist
    create_folders(output_folder, num_classes=10)

    # Iterate through the DataFrame and move images to corresponding folders
    for index, row in df.iterrows():
        image_name = row['img ']
        image_label = row['label']

        source_path = os.path.join(input_folder, image_name)
        destination_path = os.path.join(output_folder, f"{image_label}", image_name)

        shutil.copy(source_path, destination_path)

# Example usage
csv_file_path = r'C:\Users\Aryan\icip\img_label_cifar_gen.csv'
input_images_folder = r'C:\Users\Aryan\icip\fine_tune_dataset\dataset'
output_sorted_folder = r'C:\Users\Aryan\icip\fine_tune_dataset\sorted'

sort_images(csv_file_path, input_images_folder, output_sorted_folder)
