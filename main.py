# This is the App tools for style transfer and object swapping by
# using diffusion model
from config_setup import Config_setup
from UI_Control import UI_Controller
# from data_edit import Data_Edit
import resource
import os

def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

if __name__ == "__main__":

    # Setup configuration param
    config = Config_setup(False)
    # Data_Edit(config)
    UI_Controller(config)
    
    
    
    
    
