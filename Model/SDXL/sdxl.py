from diffusers import (
    DPMSolverMultistepScheduler,
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLPipeline
)
import torch
from transformers import CLIPVisionModelWithProjection



class SDXL:

    def __init__(self,model_config):

        print("Loading Diffusion model")
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
        if "Depth" in model_config["Controlnet"]:
            controlnet2_Depth = ControlNetModel.from_pretrained(self.config.model_setting["Controlnet"]["Depth"][1], 
                                                            torch_dtype=torch.float16, 
                                                            use_safetensors=True,
                                                            variant="fp16",
                                                            ).to("cuda")
            print("Load Depth")
            control.append(controlnet2_Depth)
        if "Dwpose" in model_config["Controlnet"]:
            controlnet_pose = ControlNetModel.from_pretrained(self.config.model_setting["Controlnet"]["Dwpose"][1],
                                                            torch_dtype=torch.float16,
                                                            use_safetensors=True,
                                                            ).to("cuda")
            print("Load Dwpose")
            control.append(controlnet_pose)
        if "Canny" in model_config["Controlnet"]:
            controlnet_Canny = ControlNetModel.from_pretrained(self.config.model_setting["Controlnet"]["Canny"][1], 
                                                            torch_dtype=torch.float16, 
                                                            use_safetensors=True,
                                                            variant="fp16",
                                                            ).to("cuda")
            print("Load Canny")
            control.append(controlnet_Canny)
        if "Lineart" in model_config["Controlnet"]:
            controlnet_lineart = ControlNetModel.from_pretrained(self.config.model_setting["Controlnet"]["Lineart"][1], 
                                                            torch_dtype=torch.float16, 
                                                            use_safetensors=True,
                                                            #    variant="fp16",
                                                            ).to("cuda")
            print("Load Lineart")
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
        scale = model_config["Ipadapter"][0]
        self.pipeline.load_ip_adapter(self.config.model_setting["Ipadapter"][1],subfolder="sdxl_models",weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"])
        self.pipeline.set_ip_adapter_scale([scale])

        # set param for main
        self.pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline.enable_model_cpu_offload()
        # self.vae = self.pipeline.vae
        self.generator = torch.Generator("cuda").manual_seed(92)


        if model_config["Differential"] != "":
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
