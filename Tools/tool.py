import json
from datetime import datetime
from colorama import Fore, Back, Style
import os

def info():
     return Fore.GREEN + "[Info]" + Fore.RESET + ' '

def warning():
     return Fore.YELLOW + "[Warning]"+ Fore.RESET + ' '

def critical():
     return Fore.RED + "[Critical]"+ Fore.RESET + ' '


def get_datetime(mode):
    if mode == 0:
        return datetime.now().strftime("%H%M%S")
    if mode == 1:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_json(file_path):
        #Load json
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Unable to load {file_path}: {e}")
            exit(1)

def walk_path(path):
    print(f"Walking directory tree from: {path}")
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            print(f"\nCurrent Directory: {dirpath}")
            if dirnames:
                print("  Subfolders:")
                for dirname in dirnames:
                    print(f"    - {dirname}")
            if filenames:
                print("  Files:")
                for filename in filenames:
                    print(f"    - {filename}")
    except FileNotFoundError:
        print(f"Error: Directory not found at {path}")
    except PermissionError:
        print(f"Error: Permission denied to access {path}")

def list_files_and_folders(path,pre):
    print(f"Listing contents of: {path}")
    subpath = list()
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                print(pre + f"  File: {item}")
            elif os.path.isdir(item_path):
                print(pre + f"  Folder: {item}")
                subpath.append(item_path)
        if len(subpath) != 0 : 
            pre += "-"
            for i in subpath:
                list_files_and_folders(i,pre)

    except FileNotFoundError:
        print(f"Error: Directory not found at {path}")
    except PermissionError:
        print(f"Error: Permission denied to access {path}")

if __name__ == "__main__":
    
    list_files_and_folders('./Input/Test_All',"")