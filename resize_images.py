import os
from PIL import Image

def resize_images(input_folder, output_folder, size=(256, 256)):
    """
    Resize all images in the input folder to the specified size and save them in the output folder.

    :param input_folder: Path to the folder containing the original images.
    :param output_folder: Path to the folder where resized images will be saved.
    :param size: Tuple specifying the size (width, height) of the resized images.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Check if the file is an image
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            try:
                # Open the image
                with Image.open(input_path) as img:
                    # Resize the image
                    resized_img = img.resize(size, Image.ANTIALIAS)

                    # Save the resized image to the output folder
                    output_path = os.path.join(output_folder, filename)
                    resized_img.save(output_path)

                    print(f"Resized and saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = ""
    output_folder = ""

    # Call the function to resize images
    resize_images(input_folder, output_folder)
