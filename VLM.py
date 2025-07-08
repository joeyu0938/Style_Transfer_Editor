from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import cv2
from glob import glob
from PIL import Image
import json
import numpy as np
import gc
from config_setup import Config_setup
import torch
from Tools.tool import warning,info,critical
import os
from datetime import datetime


class VLM():

    def __init__(self,config:Config_setup):
        self.config = config
        self.history = dict()
        self.prompt_dict = dict()
        self.vlm_name = self.config.model_setting["VLM"]
        self.rules = "You are a helpful assistant. If user asks for describing image, " \
        "please only output(response) in the format of json dictionary with prompt1 and prompt2 keys"

    def unload_model(self):
        del self.vlm
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()

    def load_model(self):
        #set up vlm
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.vlm_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.vlm_name,use_fast = True)
    
    def create_message(self,prompt:str,image_path=None,video_path=None,history=False):
        message = [
            {"role": "user" , "content": [
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        if image_path!= None and len(image_path)>=1:
            message.append({"role": "system", "content": f"{self.rules}"})
            if os.path.isdir(image_path):
                self.images = glob(image_path+'/*.jpg') + glob(image_path+'/*.png')
                for img in self.images :
                    message[0]["content"].append({"type": "image", "image": img})
            elif os.path.isfile(image_path):
                message[0]["content"].append({"type": "image", "image": image_path})

        if video_path != None and len(video_path)>=1:
            message.append({"role": "system", "content": f"{self.rules}"})
            message[0]["content"].append({"type": "video", "video": video_path, "max_pixels": 360 * 420 ,"fps": 8.0})
        return message

    def clear_history(self):
        self.history.clear()

    def output_history(self):
        with open(self.config.path_setting["output_folder"] + f'/prompt.json', "w") as outfile:
            json.dump(self.history, outfile, sort_keys=True, indent=2)

    def set_prompt(self,prompt:str,image_path=None):
        
        try:
            p = prompt.split('{')[1]
            p = p.split('}')[0]
            p = '{' + p + '}'
            self.prompt_dict[f"{image_path}"] = json.loads(p)
            print(info() + f"Successfully set {image_path} prompt dict ")
            print(info() + f"   -{self.prompt_dict[f"{image_path}"]["prompt1"]}")
            print(info() + f"   -{self.prompt_dict[f"{image_path}"]["prompt2"]}")
        except Exception as e:
            print(critical() + "Json dictionary(Prompt1&2) from VLM incorrect (Modified prompt and run again)")
            

    def run_vlm(self,prompt= "Hi who are you?",image_path = None,video_path= None,show_prompt=False):
        messages = self.create_message(prompt,image_path,video_path)

        if show_prompt == True:
            print(info() + messages)

        self.history[f"{datetime.now().strftime("%Y%m%d_%H%M%S")}"] = messages
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.vlm.generate(**inputs, max_new_tokens=77,temperature = 0.95,do_sample = True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if len(image_path)>=1 or len(video_path)>=1:
            self.set_prompt(output_text[0],image_path)

        return output_text