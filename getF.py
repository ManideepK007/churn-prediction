import os

def list_one_level(directory):
    print(f"Current directory: {directory}")
    print("\nFiles and folders in current directory:")
    for item in os.listdir(directory):
        print(f"  {item}")

    print("\nOne-level-down contents:")
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            print(f"\nInside folder: {item}")
            try:
                for subitem in os.listdir(full_path):
                    print(f"  {subitem}")
            except PermissionError:
                print("  [Permission Denied]")

# Change '.' to the directory you want to inspect, or leave as is for current dir
list_one_level('.')
