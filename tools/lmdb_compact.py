#!/usr/bin/env python3
import os
import lmdb


####### Usage : python lmdb_compact.py D:\parsec\digits\train D:\parsec\digits_compressed\train
####### python lmdb_compact.py D:\parsec\digits\val D:\parsec\digits_compressed\val

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


def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Compact an LMDB database.")
    parser.add_argument('input_db', help='Path to the input LMDB database.')
    parser.add_argument('output_db', help='Path to the output (compacted) LMDB database.')
    args = parser.parse_args()

    # Compact the database
    compact_lmdb(args.input_db, args.output_db)


if __name__ == '__main__':
    main()