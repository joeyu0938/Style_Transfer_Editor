# Set model for inference
#Register model
#   
#   Please provide register Model class with below function in Model class(self.Model_dict)
#   - Load_model
#   - Clear_model
#   - Run_model
import os
from Model.SDXL.sdxl import SDXL
from config_setup import Config_setup
from glob import glob
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from Tools.Image_Loader import Image_Dataset
from Tools.tool import warning,info,critical
from Tools.tool import Recover


class Model:

    def __init__(self,config:Config_setup,ui):
        
        self.config = config
        self.ui = ui
        self.log = dict()
        self.Model_dict = {
            "SDXL": SDXL
        }
    
    def load_model(self):

        # try:
        self.model = self.Model_dict[self.config.model_setting["Model"]](self.config)
        self.model.Load_model()
        # except:
        #     print(critical()+f"Fail to load Model {self.config.model_setting["Model"]}")
    
    def load_data(self):
        try:
            self.dataset = Image_Dataset(self.config)
            self.dataloader = DataLoader(self.dataset,batch_size=self.config.config_setting["batch_size"])
        except:
            print(critical()+f"Fail to load Dataset and Dataloader ")

    def run_model(self,prompt_dict,UI_Bar,lcd_number):

        ref_images = glob(self.config.config_setting["reference_img"]+'/*jpg')

        if UI_Bar != None:
            UI_Bar.setRange(0, len(self.dataloader)*len(ref_images))
            UI_Bar.setValue(0)

        if self.config.system_setting["Task"] == "img_cond_inpaint":
            for ref_idx,ref in enumerate(ref_images):
                r = Image.open(ref)
                ref_name = os.path.splitext(os.path.basename(ref))[0]

                with tqdm(self.dataloader) as pbar:
                    for i,(image,mask,cond,cond_scale,basename) in enumerate(pbar):
                        
                        if UI_Bar != None:
                            UI_Bar.setValue((ref_idx+1)*(i+1))
                        if lcd_number != None:
                            # elapsed = pbar.format_dict["elapsed"]
                            rate = pbar.format_dict["rate"]
                            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
                            remaining/=60
                            lcd_number.display('{:.02f}'.format(remaining))

                        imgs = self.model.Run_model(
                            prompt1= prompt_dict[ref]["prompt1"],
                            prompt2 = prompt_dict[ref]["prompt2"],
                            image = image,
                            mask = mask,
                            cond = cond,
                            cond_scale = cond_scale,
                            control_guidance_start = self.config.model_setting["control_guidance_start"],
                            control_guidance_end = self.config.model_setting["control_guidance_end"],
                            negprompt = self.config.config_setting["negprompt"],
                            ip_adapter_image = r,
                        )

                        for idx,img in enumerate(imgs):
                            if self.config.config_setting["Is_crop"] == True:
                                img.save(self.config.path_setting["output_folder"]+'/Tmp/crop_output/'+ basename[idx])
                            else:
                                img.save(self.config.path_setting["output_folder"]+'/images/'+ basename[idx])

                if self.config.config_setting["Is_crop"] == True:
                    Recover(self.config.path_setting["output_folder"],ref_name)

