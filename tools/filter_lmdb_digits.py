#!/usr/bin/env python3
import io
import os
import shutil
from argparse import ArgumentParser

import lmdb
import numpy as np
from PIL import Image


def is_digit_label(label):
    """Check if the label contains only digits."""
    if label is None:
        return False
    return label.decode('utf-8').isdigit()
    
def find_lmdb_files(directory):
    """Find all .mdb files in the given directory (recursively)."""
    lmdb_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.mdb'):
                lmdb_files.append(os.path.join(root, file))
    return lmdb_files
    
def compact_lmdb(input_db, output_db):
    """
    Compact an LMDB database by copying all entries to a new database.
    
    Args:
        input_db (str): Path to the input LMDB database.
        output_db (str): Path to the output (compacted) LMDB database.
    """
    # Open the input LMDB database
    with lmdb.open(input_db, readonly=True) as env_in:
        # Get the map_size of the input database
        map_size = env_in.info()['map_size']
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_db, exist_ok=True)
        
        # Open the output LMDB database
        with lmdb.open(output_db, map_size=map_size) as env_out:
            # Copy all entries from the input to the output database
            with env_in.begin() as txn_in:
                with env_out.begin(write=True) as txn_out:
                    cursor = txn_in.cursor()
                    for key, value in cursor:
                        txn_out.put(key, value)
    
    print(f"Database compaction completed. Compacted database saved to: {output_db}")
    
    

#### usage : python filter_lmdb_digits.py D:\parsec\input --output D:\parsec\test

# Train : D:\parsec\data\MJ_train\train\synth\MJ\train D:\parsec\data\train
# Val : D:\parsec\data\MJ_val\train\synth\MJ\val D:\parsec\data\val

#### usage : python filter_lmdb_digits.py D:\parsec\data\MJ_train\train\synth\MJ\train D:\parsec\data\train --output D:\parsec\digits\train

# Skipping: 714770, label is not digits
# Written samples from 713769 to 714769
# Written 69794 samples to D:\parsec\digits\train out of 7939356 input samples.


#### usage : python filter_lmdb_digits.py D:\parsec\data\MJ_val\train\synth\MJ\val D:\parsec\data\val --output D:\parsec\digits\val


def main():
    parser = ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='Path to input LMDBs')
    parser.add_argument('--output', help='Path to output LMDB')
    parser.add_argument('--min_image_dim', type=int, default=8)
    args = parser.parse_args()

    # Find all LMDB files in the input directories
    lmdb_files = []
    for input_path in args.inputs:
        if os.path.isdir(input_path):
            lmdb_files.extend(find_lmdb_files(input_path))
        elif os.path.isfile(input_path) and input_path.endswith('.mdb'):
            lmdb_files.append(input_path)
        else:
            print(f"Error: Invalid input path (not a directory or .mdb file): {input_path}")
            return

    # Verify that at least one LMDB file was found
    if not lmdb_files:
        print("Error: No LMDB files found in the input directories.")
        return

    # Calculate the total size of the LMDB files
    input_size = sum(os.path.getsize(lmdb_file) for lmdb_file in lmdb_files)
    map_size = input_size * 1  # Set map_size to factor x
    print(f"Total size of LMDB files: {input_size} bytes")
    print(f"Setting map_size to: {map_size} bytes")

    os.makedirs(args.output, exist_ok=True)
    temp_output = args.output + "_temp"
    with lmdb.open(temp_output, map_size=map_size) as env_out:
    # with lmdb.open(temp_output, map_size=map_size, writemap=True) as env_out:
        in_samples = 0
        out_samples = 0
        samples_per_chunk = 1000

        # Set to track unique indices
        unique_indices = set()

        for lmdb_in in args.inputs:
            print(f"Processing input LMDB: {lmdb_in}")
            try:
                with lmdb.open(lmdb_in, readonly=True, max_readers=1, lock=False) as env_in:
                    print("LMDB file opened successfully.")
                    with env_in.begin() as txn:
                        # Dynamically calculate num_samples if the key is missing
                        num_samples_key = 'num-samples'.encode()
                        num_samples = txn.get(num_samples_key)
                        if num_samples is None:
                            print("'num-samples' key not found. Calculating dynamically...")
                            num_samples = 0
                            cursor = txn.cursor()
                            for key, _ in cursor:
                                if key.startswith(b'image-'):
                                    num_samples += 1
                        else:
                            num_samples = int(num_samples)
                        print(f"Total samples in LMDB: {num_samples}")
                    in_samples += num_samples
                    chunks = np.array_split(range(num_samples), num_samples // samples_per_chunk)
                    for chunk in chunks:
                        cache = {}
                        with env_in.begin() as txn:
                            for index in chunk:
                                index += 1  # lmdb starts at 1
                                image_key = f'image-{index:09d}'.encode()
                                label_key = f'label-{index:09d}'.encode()

                                print(f"Processing index: {index}")
                                print(f"Image key: {image_key.decode('utf-8')}")
                                print(f"Label key: {label_key.decode('utf-8')}")

                                # Fetch image and label
                                image_bin = txn.get(image_key)
                                label_bin = txn.get(label_key)

                                # Skip if image or label is missing
                                if image_bin is None:
                                    print(f'Skipping: {index}, missing image')
                                    continue
                                if label_bin is None:
                                    print(f'Skipping: {index}, missing label')
                                    continue

                                # Skip if label does not contain only digits
                                if not is_digit_label(label_bin):
                                    print(f'Skipping: {index}, label is not digits')
                                    continue

                                # Check image dimensions
                                try:
                                    img = Image.open(io.BytesIO(image_bin))
                                    w, h = img.size
                                    print(f"Image dimensions: {w}x{h}")
                                    if w < args.min_image_dim or h < args.min_image_dim:
                                        print(f'Skipping: {index}, w = {w}, h = {h}')
                                        continue
                                except Exception as e:
                                    print(f"Error processing image: {e}")
                                    continue

                                # Skip if the index is already in the output
                                if index in unique_indices:
                                    print(f'Skipping: {index}, duplicate index')
                                    continue

                                # Add the index to the unique set
                                unique_indices.add(index)

                                # If all checks pass, add to output
                                out_samples += 1  # increment. start at 1
                                out_image_key = f'image-{out_samples:09d}'.encode()
                                out_label_key = f'label-{out_samples:09d}'.encode()
                                cache[out_image_key] = image_bin
                                cache[out_label_key] = label_bin
                        with env_out.begin(write=True) as txn:
                            for k, v in cache.items():
                                txn.put(k, v)
                        print(f'Written samples from {chunk[0]} to {chunk[-1]}')
            except Exception as e:
                print(f"Error processing LMDB file {lmdb_in}: {e}")
                continue

        with env_out.begin(write=True) as txn:
            txn.put('num-samples'.encode(), str(out_samples).encode())
        print(f'Written {out_samples} samples to {temp_output} out of {in_samples} input samples.')

            
    # Compact the database
    print("Compacting the output LMDB database...")
    compact_lmdb(temp_output, args.output)
    # print("Database compaction completed successfully.")

    # Clean up the temporary database
    print(f"Deleting temporary database: {temp_output}")
    shutil.rmtree(temp_output)
    # print("Temporary database deleted successfully.")

if __name__ == '__main__':
    main()