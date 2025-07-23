from torch.utils.data import Dataset
from config_setup import Config_setup
from glob import glob
import os
from torchvision import transforms
import torch
from PIL import Image
from Tools.tool import critical,info,warning

class Image_Dataset(Dataset):


    def globandsort(self,path):
        path = glob(path)
        path = sorted(path)
        return path

    def __init__(self,config:Config_setup,transform=None):

        self.config = config
        print(info() + "Initiating Image Dataset...")
        controlnet_cnt = 0 

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(), # Converts PIL Image to torch.FloatTensor (0-1 range)
                # Add other transformations here, e.g., Normalize, Resize, etc.
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Example: normalize to [-1, 1]
            ])


        self.image_path = self.globandsort(self.config.path_setting["Image_folder"]+"/*.jpg")
        self.mask_path = self.globandsort(self.config.path_setting["Mask_folder"]+"/*.jpg")
        # By order
        # Depth 
        # Canny 
        # Dwpose 
        # Lineart
        if "Depth" in self.config.model_setting["Controlnet"]:
            p = self.config.path_setting["output_folder"] + "/Tmp/depth"
            self.depth_path = self.globandsort(p+'/*.jpg')
            print(info()+f"Find Depth in Contolnet,load {p} to dataset path ")
            controlnet_cnt+=1
        else:
            print(warning()+ "No Depth in Controlnet")
        if "Canny" in self.config.model_setting["Controlnet"]:
            p = self.config.path_setting["output_folder"] + "/Tmp/canny"
            self.canny_path = self.globandsort(p+'/*.jpg')
            print(info()+ f"Find Canny in Contolnet,load {p} to dataset path ")
            controlnet_cnt+=1
        else:
            print(warning()+ "No Canny in Controlnet")
        if "Dwpose" in self.config.model_setting["Controlnet"]:
            p = self.config.path_setting["output_folder"] + "/Tmp/dwpose"
            self.dwpose_path = self.globandsort(p+'/*.jpg')
            print(info()+ f"Find Dwpose in Contolnet,load {p} to dataset path ")
            controlnet_cnt+=1
        else:
            print(warning()+ "No Dwpose in Controlnet")
        if "Lineart" in self.config.model_setting["Controlnet"]:
            p = self.config.path_setting["output_folder"] + "/Tmp/lineart"
            self.lineart_path = self.globandsort(p+'/*.jpg')
            print(info()+ f"Find Lineart in Contolnet,load {p} to dataset path ")
            controlnet_cnt+=1
        else:
            print(warning()+ "No Lineart in Controlnet")

        print(info() + f" Total => {controlnet_cnt} type of contol :file to Load")
    
    def __getitem__(self, index):

        image = Image.open(self.image_path[index])
        image = self.transform(image)

        mask = Image.open(self.mask_path[index])
        mask = self.transform(mask)

        cond = list()
        cond_scale = list()

        if "Depth" in self.config.model_setting["Controlnet"]:
            depth = Image.open(self.depth_path[index])
            depth = self.transform(depth)
            cond.append(depth)
            cond_scale.append(self.config.model_setting["Controlnet"]["Depth"][0])
        if "Dwpose" in self.config.model_setting["Controlnet"]:
            openpose = Image.open(self.dwpose_path[index])
            openpose = self.transform(openpose)
            cond.append(openpose)
            cond_scale.append(self.config.model_setting["Controlnet"]["Dwpose"][0])
        if "Canny" in self.config.model_setting["Controlnet"]:
            canny = Image.open(self.canny_path[index])
            canny = self.transform(canny)
            cond.append(canny)
            cond_scale.append(self.config.model_setting["Controlnet"]["Canny"][0])
        if "Lineart" in self.config.model_setting["Controlnet"]:
            lineart = Image.open(self.lineart_path[index])
            lineart = self.transform(lineart)
            cond.append(lineart)
            cond_scale.append(self.config.model_setting["Controlnet"]["Lineart"][0])

        cond_scale = torch.tensor(cond_scale).to('cuda')
        print(cond_scale.shape)

        return image,mask,cond,cond_scale,os.path.basename(self.image_path[index])
    
    def __len__(self):
        
        return len(self.image_path)