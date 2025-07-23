from diffusers import (
    DPMSolverMultistepScheduler,
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLPipeline
)
import torch
from transformers import CLIPVisionModelWithProjection
from config_setup import Config_setup
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms
import gc 
import os
from Tools.tool import warning,info,critical

def load_image_path(path):
    files = glob(f'{path}/*.jpg')
    print(info()+f" Loading crop file from {path}...")
    sorted_files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return sorted_files

class SDXL:
    


    def __init__(self,config:Config_setup):
        self.config = config
        self.generator = torch.Generator(device="cuda").manual_seed(30)

    def Load_model(self):
        print(info()+" Loading SDXL Diffusion model")
        #Load sdxl
        sdxl = self.config.model_setting["SDXL"]

        #  #Load scheduler
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            sdxl,
            subfolder="scheduler",
            algorithm_type = "sde-dpmsolver++",
            # clip_sample=False,
            # use_karras_sigmas = True,
            # timestep_spacing="linspace",
            # beta_schedule="linear",
            use_safetensors=True,
        )

        #Load clipvision
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.config.model_setting["Ipadapter"][1],
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        control = list()

        # #Set Multi-controlnet
        if "Depth" in self.config.model_setting["Controlnet"]:
            controlnet2_Depth = ControlNetModel.from_pretrained(self.config.model_setting["Controlnet"]["Depth"][1], 
                                                            torch_dtype=torch.float16, 
                                                            use_safetensors=True,
                                                            variant="fp16",
                                                            ).to("cuda")
            print("Load Depth controlnet")
            control.append(controlnet2_Depth)
        if "Dwpose" in self.config.model_setting["Controlnet"]:
            controlnet_pose = ControlNetModel.from_pretrained(self.config.model_setting["Controlnet"]["Dwpose"][1],
                                                            torch_dtype=torch.float16,
                                                            use_safetensors=True,
                                                            ).to("cuda")
            print("Load Dwpose controlnet")
            control.append(controlnet_pose)
        if "Canny" in self.config.model_setting["Controlnet"]:
            controlnet_Canny = ControlNetModel.from_pretrained(self.config.model_setting["Controlnet"]["Canny"][1], 
                                                            torch_dtype=torch.float16, 
                                                            use_safetensors=True,
                                                            variant="fp16",
                                                            ).to("cuda")
            print("Load Canny controlnet")
            control.append(controlnet_Canny)
        if "Lineart" in self.config.model_setting["Controlnet"]:
            controlnet_lineart = ControlNetModel.from_pretrained(self.config.model_setting["Controlnet"]["Lineart"][1], 
                                                            torch_dtype=torch.float16, 
                                                            use_safetensors=True,
                                                            #    variant="fp16",
                                                            ).to("cuda")
            print("Load Lineart controlnet")
            control.append(controlnet_lineart)

        self.pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            sdxl,
            # unet = unet,
            # vae = self.vae,
            controlnet = control,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            image_encoder = image_encoder,
            scheduler = scheduler,
        ).to("cuda")
        
         # self.pipeline.unet = torch.compile(self.pipeline.unet, mode="reduce-overhead", fullgraph=True)

        #Load ipadapter
        scale = self.config.model_setting["Ipadapter"][0]
        self.pipeline.load_ip_adapter(self.config.model_setting["Ipadapter"][1],subfolder="sdxl_models",weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"])
        self.pipeline.set_ip_adapter_scale([scale])

        # set param for main
        self.pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline.enable_model_cpu_offload()
        # self.vae = self.pipeline.vae
        self.generator = torch.Generator("cuda").manual_seed(92)


        if self.config.model_setting["Differential"] == True:
            print(info()+"Using SDXL Differential diffusion")
            self.different = StableDiffusionXLPipeline.from_pretrained(
                sdxl,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                image_encoder = image_encoder,
                scheduler = scheduler,
                custom_pipeline="pipeline_stable_diffusion_xl_differential_img2img",
            )
            scale = self.config.model_setting["Ipadapter"][0]
            self.different.load_ip_adapter(self.config.model_setting["Ipadapter"][1],subfolder="sdxl_models",weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"])
            self.different.set_ip_adapter_scale([0.8])
            self.different.enable_xformers_memory_efficient_attention()
            self.different.enable_model_cpu_offload()
        
        self.pipeline.set_progress_bar_config(disable=True)
        self.different.set_progress_bar_config(disable=True)

    def Clear_model(self):
        if hasattr(self,"pipeline"):
            del self.pipeline
        if hasattr(self,"different"):
            del self.different
        gc.collect()
        torch.cuda.empty_cache()

    def preprocess_image(self,image):
        image = image * 2 - 1

        if image.shape[0] != 3:
            image = np.transpose(image, (2, 0, 1))

        if len(image.shape) == 3:
            image = image[np.newaxis,...]

        return torch.tensor(image).to('cuda')
    
    def preprocess_map(self,map):
        # map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
        # convert to tensor
        map = 1 - map[0]
        map = map.to("cuda")
        return map

    def Run_model(self,prompt1,prompt2,image,mask,cond,control_guidance_start,control_guidance_end,ip_adapter_image,negprompt,cond_scale):
        
        image_ten = self.pipeline(
                        prompt = prompt1,
                        prompt_2 = prompt2,
                        image = image,
                        mask_image = mask,
                        control_image= cond, 
                        control_guidance_start = control_guidance_start, 
                        control_guidance_end = control_guidance_end,
                        ip_adapter_image= ip_adapter_image,
                        strength = 1.0,
                        negative_prompt= negprompt,
                        generator = self.generator,
                        num_inference_steps= 8,
                        guidance_scale= 2.5,
                        output_type = "np",
                        controlnet_conditioning_scale= cond_scale[0],
                        ).images[0]
        
        image_ten = self.preprocess_image(image_ten)
        mask_ten = self.preprocess_map(mask)

        result = self.different(
                    prompt = prompt1,
                    prompt_2 = prompt2,
                    negative_prompt=negprompt,
                    generator = self.generator,
                    width=1024,
                    height=1024,
                    guidance_scale= 1.5,
                    num_inference_steps = 8,
                    original_image= image_ten,
                    ip_adapter_image= ip_adapter_image,
                    image = image_ten,
                    # strength= 0.8,
                    denoising_start = 0.8,
                    map=mask_ten,
                ).images
        
        return result