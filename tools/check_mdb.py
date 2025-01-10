import lmdb
import pickle
import os

# Path to the LMDB directory (not the file itself)
lmdb_path = r'D:\parsec\data\val'

# Verify the path exists
if not os.path.exists(lmdb_path):
    print(f"Error: The path '{lmdb_path}' does not exist.")
    exit(1)

# Open the LMDB file in read-only mode
try:
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
except lmdb.Error as e:
    print(f"Error opening LMDB database: {e}")
    exit(1)
    
# Set to store unique keys
# column_headers = set()

# Iterate through the database and collect keys
# with env.begin() as txn:
    # cursor = txn.cursor()
    # for key, _ in cursor:  # We only care about the keys, not the values
        # column_headers.add(key.decode('utf-8'))  # Decode bytes to string
        
# Print the column headers
# print("Column Headers:")
# for header in column_headers:
    # print(header)
    
    
# Fetch the first key-value pair
with env.begin() as txn:
    cursor = txn.cursor()
    if cursor.first():  # Move to the first entry
        key = cursor.key()
        value = cursor.value()

        # Decode the key and deserialize the value (if necessary)
        key_str = key.decode('utf-8')  # Decode bytes to string
        try:
            value_data = pickle.loads(value)  # Deserialize the value
        except pickle.UnpicklingError:
            value_data = value  # Use raw value if deserialization fails

        # Print the first entry
        print("First Entry:")
        print(f"Key: {key_str}")
        print(f"Value: {value_data}")
    else:
        print("The database is empty.")

# Function to deserialize the value (if needed)
# def deserialize_value(value):
    # try:
        # return pickle.loads(value)  # Assuming the value is a serialized object
    # except pickle.UnpicklingError:
        # return value  # Return the raw value if deserialization fails

# Iterate through the database and print keys and values
# with env.begin() as txn:
    # cursor = txn.cursor()
    # for key, value in cursor:
        # print(f"Key: {key}")
        # deserialized_value = deserialize_value(value)
        # print(f"Value: {deserialized_value}")
        # print("-" * 40)  # Separator for readability

# Close the LMDB environment
env.close()