# This is the App tools for style transfer and object swapping by
# using diffusion model
from config_setup import Config_setup
from UI_Control import UI_Controller

if __name__ == "__main__":

    # Setup configuration param
    config = Config_setup(False)
    UI_Controller(config)
    
    
    
    
