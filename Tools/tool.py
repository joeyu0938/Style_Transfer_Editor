import json
from datetime import datetime
from colorama import Fore, Back, Style
import os
import cv2
from tqdm import tqdm
import shutil
from glob import glob

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

def Recover(path,folder_name):
        
        json_path = path + '/Tmp/Recover.json'
        crop_imgs = path + '/Tmp/crop_output'
        crop_msks = path + '/Tmp/Mask'
        os.mkdir(path +'/images/' + folder_name)
        output_path = path  + '/images/' + folder_name
        

        print(info() + "Recovering cropped images....")
        if not os.path.exists(json_path):
            print(critical() + "No Recover json found")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            recover_list = json.load(f)

        for key in recover_list.keys():
            origin = cv2.imread(key)
            name = os.path.basename(key)
            for img in recover_list[key].keys():
                print(f"Recovering {img}")
                x_min,y_min,x_max,y_max = recover_list[key][img][0],recover_list[key][img][1],recover_list[key][img][2],recover_list[key][img][3]
                w = x_max-x_min
                h = y_max-y_min
                crop_img = cv2.imread(crop_imgs + f'/{img}')
                crop_img = cv2.resize(crop_img,(w,h))
                crop_msk = cv2.imread(crop_msks + f'/{img}')
                crop_msk = cv2.resize(crop_msk,(w,h))/255
                pos = recover_list[key][img]
                crop_img = crop_img*crop_msk + origin[y_min:y_max,x_min:x_max:]*(1-crop_msk)
                origin[y_min:y_max,x_min:x_max:] = crop_img

            cv2.imwrite(f'{output_path}/{name}',origin)

        print(info() + "Finish recovering")


if __name__ == "__main__":
    
    list_files_and_folders('./Input/Test_All',"")