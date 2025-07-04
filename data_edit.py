import cv2
from glob import glob
import os
import shutil
import imageio
import torch
from dwpose import DwposeDetector
import gc
from config_setup import Config_setup
import numpy as np
from tqdm import tqdm
from controlnet_aux import MidasDetector,LineartDetector, CannyDetector
from config_setup import Config_setup
from datetime import datetime
from Tools import tool
from PIL import Image
from Tools.tool import warning,info,critical
import json


class Data_Edit:

    def __init__(self,config:Config_setup):
        self.config = config
        self.Setup_Output()
        self.Reset_Tmp()
    
    def Setup_Output(self,mode="Create"):
        '''
            Check path, delete old , create new
            
            Args:
                path_config: config path setting
            
        ''' 
        if mode == "Create":
            output_folder = datetime.now().strftime("%Y%m%d_%H%M%S")

            if len(self.config.path_setting["output_folder"]) != 0:
                output = self.config.path_setting["output_folder"]
                if os.path.exists(output):
                    shutil.rmtree(output)
            else:
                output = './Output/'+ f'{output_folder}'
            os.makedirs(f'{output}')
            os.makedirs(f'{output}/videos')
            os.makedirs(f'{output}/images')
            os.makedirs(f'{output}/images_recover')
            os.makedirs(f'{output}/images_mix')
            os.makedirs(f'{output}/reference')
            os.makedirs(f'{output}/Tmp')
            print("Finish create all needed output folder")
            self.config.path_setting["output_folder"] = output
            self.config.update(path_setting=self.config.path_setting)
            
            return f'{output}'
        
        if mode == "Destroy":
            path = os.walk('./Output')
            for i in path:
                shutil.rmtree(i[0])
            os.makedirs('./Output')
            print("Clear all subfolder under Tmp folder")
            exit()

    def Reset_Tmp(self):
        '''
            Check path, delete old , create new
            
            Args:
                path_config: config path setting
        
        '''
        for path in self.config.path_setting["Tmp_paths"]:
            os.makedirs(self.config.path_setting["output_folder"] + path)
        print(info() + " Finish create Tmp subfolders")
        pass 

    def Auto_run(self):
        
        pass

    def Dilate(self,images,kernal=1,iter=1,gaussian=1,UI_Bar=None):
        
        if UI_Bar != None:
            UI_Bar.setRange(0, len(images)-1)
            UI_Bar.setValue(0)

        for idx in tqdm(range(len(images))):
            if UI_Bar != None:
                UI_Bar.setValue(idx)
            msk = cv2.imread(images[idx])
            name = os.path.basename(images[idx])
            msk = cv2.cvtColor(msk,cv2.COLOR_BGR2GRAY)
            ret, output = cv2.threshold(msk,10,255,cv2.THRESH_BINARY)
            msk = cv2.dilate(output,np.ones((kernal,kernal), np.uint8), iterations = iter)
            msk = cv2.GaussianBlur(msk,(gaussian,gaussian), 0) 
            cv2.imwrite(self.config.path_setting["output_folder"] + f'/Tmp/Mask/{name}',msk)

    def expand_bbox(self,x_min, x_max, y_min, y_max, scale, img_shape,crop_width,crop_height):
        h, w = img_shape[:2]
        # print(f'{x_min}, {x_max}, {y_min}, {y_max}')
        # Compute center
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        crop_height = (x_max - x_min)
        crop_width = (y_max - y_min)
        crop = max(crop_height,crop_width)
        # # Method 1 : compute crop_width * crop_height and scale box
        half_h = crop * scale / 2
        half_w = crop * scale / 2
        new_x_min = int(max(0, cx - half_w))
        new_x_max = int(min(w, cx + half_w))
        new_y_min = int(max(0, cy - half_h))
        new_y_max = int(min(h, cy + half_h))

        if new_x_min > x_min : new_x_min = x_min
        if new_x_max < x_max : new_x_max = x_max
        if new_y_min > y_min : new_y_min = y_min
        if new_y_max < y_max : new_y_max = y_max
        
        # print(f'{new_x_min}, {new_x_max}, {new_y_min}, {new_y_max}')
        if (new_x_max-new_x_min) != (new_y_max-new_y_min):
            # print("correcting")
            if (new_x_max-new_x_min) <  (new_y_max-new_y_min):
                correction = (new_y_max-new_y_min)
                new_x_max = new_x_min+correction
                if new_x_max > w: 
                    new_x_max = w
                    new_x_min = new_x_max-correction
                    if new_x_min < 0: new_x_min=0
            else:
                correction = (new_x_max-new_x_min)
                new_y_max = new_y_min+correction
                if new_y_max > h: 
                    new_y_max = h
                    new_y_min = new_y_max-correction
                    if new_y_min < 0: new_y_min=0
        return new_x_min, new_x_max, new_y_min, new_y_max

    def Crop(self,images,masks,crop_scale=1.5,crop_width=1024,crop_height=512,UI_Bar = None):
        
        if UI_Bar != None:
            UI_Bar.setRange(0, len(images)-1)
            UI_Bar.setValue(0)

        print(info() + "Cropping masks and images")
        
        resolution = self.config.config_setting["resolution_formod"]
        if_crop = self.config.data_setting["if_crop"]
        json_dict = dict()

        for i in tqdm(range(len(masks))):
            color_arr = dict()
            if UI_Bar != None:
                UI_Bar.setValue(i)
                
            name = os.path.basename(images[i])
            
            if if_crop == True:
                mask = cv2.imread(masks[i])
                img = cv2.imread(images[i])
                color_np = np.unique(mask.reshape(-1,mask.shape[2]),axis=0)
                cnt = 0
                for color in color_np:
                    if list(color)== [0,0,0]: continue
                    # print(color)
                    pos= np.array(np.where((mask[:,:,0]==color[0]) & (mask[:,:,1]==color[1]) & (mask[:,:,2]==color[2])))
                    pos_min = pos.min(1)
                    pos_max = pos.max(1)
                    y_min,x_min = pos_min[0],pos_min[1]
                    y_max,x_max = pos_max[0],pos_max[1]
                    x_min, x_max, y_min, y_max = self.expand_bbox(x_min,
                                                              x_max,
                                                              y_min,
                                                              y_max,
                                                              crop_scale,
                                                              mask.shape,
                                                              crop_width,
                                                              crop_height)
                    
                    save_mask = mask[y_min:y_max,x_min:x_max,:]
                    save_img = img[y_min:y_max,x_min:x_max,:]
                    zero = np.zeros_like(save_mask)[:,:,0]
                    full = np.ones_like(save_mask)[:,:,0] * 255
                    save_mask = np.where((save_mask[:,:,0]==color[0]) & (save_mask[:,:,1]==color[1]) & (save_mask[:,:,2]==color[2]),full,zero)
                    save_mask = np.expand_dims(save_mask,axis=-1)
                    save_mask = cv2.resize(save_mask,(resolution,resolution))
                    save_img = cv2.resize(save_img,(resolution,resolution))
                    cv2.imwrite(f"{self.config.path_setting["output_folder"]}/Tmp/Mask/{name}_{cnt:05d}.jpg",save_mask)
                    cv2.imwrite(f"{self.config.path_setting["output_folder"]}/Tmp/Image/{name}_{cnt:05d}.jpg",save_img)
                    color_arr[f'{name}_{cnt:05d}.jpg'] = [x_min,y_min,x_max,y_max]
                    cnt+=1
                    # print(f"Images {i} , mask_position : min=> {x_min} {y_min} max=> {x_max} {y_max}\n")
                json_dict[images[i]] = color_arr

            else:
                mask = cv2.imread(masks[i])
                img = cv2.imread(images[i])
                save_mask = cv2.resize(mask,(resolution,resolution))
                save_img = cv2.resize(img,(resolution,resolution))
                cv2.imwrite(f"{self.config.path_setting["output_folder"]}/Tmp/Mask/{name}.jpg",save_mask)
                cv2.imwrite(f"{self.config.path_setting["output_folder"]}/Tmp/Image/{name}.jpg",save_img)
                color_arr[f'{name}.jpg'] = [0,0,resolution,resolution]
                json_dict[images[i]] = color_arr
        
        with open(self.config.path_setting["output_folder"] + f'/Tmp/Recover.json', "w") as outfile:
            json.dump(json_dict, outfile, sort_keys=True, indent=2)

    def Inpaint(self,images,masks,radius=1.0,UI_Bar=None):

        if UI_Bar != None:
            UI_Bar.setRange(0, len(images)-1)
            UI_Bar.setValue(0)

        for idx in tqdm(range(len(images))):
            if UI_Bar != None:
                UI_Bar.setValue(idx)
            img = cv2.imread(images[idx])
            msk = cv2.imread(masks[idx])
            name = os.path.basename(images[idx])
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
            msk = msk.astype(np.uint8)
            img = img.astype(np.uint8)
            dst = cv2.inpaint(img, msk, radius, cv2.INPAINT_NS)
            cv2.imwrite(self.config.path_setting["output_folder"] + f'/Tmp/Image/{name}',dst)

    def Canny(self,images,sigma=0.3,detect_res=512,image_res=512,low_thres=70,high_thres=140,UI_Bar=None):
        if UI_Bar != None:
            UI_Bar.setRange(0, len(images)-1)
            UI_Bar.setValue(0)

        try: 
            self.canny = CannyDetector()
            print(info() + "Successfully loaded canny")
        except:
            print(critical() + "Failed loading midas_depth")

        for idx in tqdm(range(len(images))):

            if UI_Bar != None:
                UI_Bar.setValue(idx)
            img = cv2.imread(images[idx])
            name = os.path.basename(images[idx])
            v = np.median(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
            l = int(max(0, (1.0 - sigma) * v))
            u = int(min(255, (1.0 + sigma) * v))
            canny_img = self.canny(
                img,low_threshold=low_thres,high_threshold=high_thres,detect_resolution=detect_res,image_resolution=image_res
            )
            cv2.imwrite(self.config.path_setting["output_folder"] + f'/Tmp/canny/{name}',canny_img)

        del self.canny
        gc.collect()

    def Dwpose(self,images,hand:bool,body:bool,face:bool,detect_res=512,image_res=512,UI_Bar=None):
        if UI_Bar != None:
            UI_Bar.setRange(0, len(images)-1)
            UI_Bar.setValue(0)
        try: 
            self.dwpose  = DwposeDetector.from_pretrained_default()
            print(info() + "Successfully loaded dwpose")
        except:
            print(critical() + "Failed loading dwpose")
        
        for idx in tqdm(range(len(images))):
            if UI_Bar != None:
                UI_Bar.setValue(idx)
            img = cv2.imread(images[idx])
            name = os.path.basename(images[idx])
            img = self.dwpose(
                    img, detect_resolution=detect_res, image_resolution=image_res,include_hand=hand,include_body=body,include_face=face
                )
            img.save(self.config.path_setting["output_folder"] + f'/Tmp/dwpose/{name}')
        
        del self.dwpose
        gc.collect()
        torch.cuda.empty_cache()

    def Lineart(self,images,detect_res=512,image_res=512,UI_Bar=None):

        if UI_Bar != None:
            UI_Bar.setRange(0, len(images)-1)
            UI_Bar.setValue(0)

        try: 
            self.lineart  = LineartDetector.from_pretrained("lllyasviel/Annotators").to(self.config.system_setting["Device"])
            print(info() + "Successfully loaded lineart")
        except:
            print(critical() + "Failed loading lineart")

        
        for idx in tqdm(range(len(images))):
            if UI_Bar != None:
                UI_Bar.setValue(idx)
            img = cv2.imread(images[idx])
            name = os.path.basename(images[idx])
            lineart = self.lineart(
                        img, detect_resolution=detect_res, image_resolution=image_res
                    )
            lineart.save(self.config.path_setting["output_folder"] + f'/Tmp/lineart/{name}')
        
        del self.lineart
        gc.collect()
        torch.cuda.empty_cache()

    def Depth(self,images,detect_res=512,image_res=512,UI_Bar=None):

        if UI_Bar != None:
            UI_Bar.setRange(0, len(images)-1)
            UI_Bar.setValue(0)
  
        try: 
            self.midas_depth  = MidasDetector.from_pretrained("lllyasviel/Annotators").to(self.config.system_setting["Device"])
            print(info() + "Successfully loaded midas_depth")
        except:
            print(critical() + "Failed loading midas_depth")
        
        for idx in tqdm(range(len(images))):
            if UI_Bar != None:
                UI_Bar.setValue(idx)
            img = cv2.imread(images[idx])
            name = os.path.basename(images[idx])
            depth = self.midas_depth(
                        img, detect_resolution=detect_res, image_resolution=image_res
                    )
            cv2.imwrite(self.config.path_setting["output_folder"] + f'/Tmp/depth/{name}',depth)
        
        del self.midas_depth
        gc.collect()
        torch.cuda.empty_cache()
        