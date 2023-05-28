import os

def delete_files_with_pattern(root_dir, pattern):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if pattern in filename:
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

# Specify the root directory and the pattern to search for in the filenames
root_directory = "."
file_pattern = "(1)"

# Call the function to delete files with the specified pattern
delete_files_with_pattern(root_directory, file_pattern)
