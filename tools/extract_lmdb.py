import os
import argparse
import lmdb

######## Usage : python extract_lmdb.py --lmdb_path D:\parsec\digits\val --output_dir  D:\parsec\extracted\val
######## python extract_lmdb.py --lmdb_path D:\parsec\digits\train --output_dir  D:\parsec\extracted\train

def extract_images_from_lmdb(lmdb_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Path for the text file
    text_file_path = os.path.join(output_dir, 'labels.txt')

    # Open the LMDB environment
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    # Open a text file to write the image names and labels
    with open(text_file_path, 'w', encoding='utf-8') as text_file:  # Use UTF-8 encoding
        # Start a new transaction
        with env.begin() as txn:
            # Iterate over all possible indices
            index = 1
            while True:
                # Construct the image and label keys
                image_key = f'image-{index:09d}'.encode()
                label_key = f'label-{index:09d}'.encode()

                # Check if the image key exists in the database
                image_data = txn.get(image_key)
                if image_data is None:
                    break  # Stop if no more images are found

                # Get the label data
                label_data = txn.get(label_key)
                if label_data is None:
                    print(f"Warning: No label found for image {image_key.decode('utf-8')}. Skipping.")
                    index += 1
                    continue

                # Decode the label
                label = label_data.decode('utf-8')

                # Format the image name as "000000001.jpg"
                image_name = f"{index:09d}.jpg"

                # Save the image to the output directory
                image_path = os.path.join(output_dir, image_name)
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_data)

                # Write the image name and label to the text file
                text_file.write(f"{image_name} {label}\n")

                # Increment the index
                index += 1

    # Close the LMDB environment
    env.close()

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Extract images and labels from an LMDB database.")
    parser.add_argument('--lmdb_path', type=str, required=True, help="Path to the input LMDB database.")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory for images and labels.")
    args = parser.parse_args()

    # Call the extraction function
    extract_images_from_lmdb(args.lmdb_path, args.output_dir)
    print(f"Extraction complete. Images and labels saved in {args.output_dir}.")

if __name__ == "__main__":
    main()