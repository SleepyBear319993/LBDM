import os
import shutil

def clean_bin_directory():
    """Remove all files from the bin directory"""
    bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
    
    if os.path.exists(bin_dir):
        print(f"Cleaning directory: {bin_dir}")
        
        # Option 1: Remove entire directory and recreate it
        shutil.rmtree(bin_dir)
        os.makedirs(bin_dir)
        
        print("Bin directory cleaned successfully")
    else:
        print("Bin directory doesn't exist")

if __name__ == "__main__":
    clean_bin_directory()