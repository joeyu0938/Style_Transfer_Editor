import argparse
import json
import Tools.tool as tools
from Tools.tool import warning,info,critical

class Config_setup:


    def __init__(self,if_show:bool):
        '''
            Setup base config ,model config, path config
            
            Args:
                if_show: " Whether to show the loaded config setting "

            Returns:
                Config_setup object
        '''
        #Parser setup
        parser = argparse.ArgumentParser()
        parser.add_argument("--Setting", type=str, default= './Config/config.json', help="Base config setting")
        self.args = parser.parse_args()

        self.Set_json(self.args.Setting,if_show)

    def reload(self,if_show=False):
        self.Set_json(self.Set_json(self.args.Setting,if_show))

    def Set_json(self,json_file,if_show):
        # Load json
        print(f"Loading json => {json_file}...")
        
        config = tools.load_json(json_file)
        

        if if_show:
            print(f"Get json:\n {json.dumps(config, indent=4, ensure_ascii=False)}")


        #Check exist
        checklist = ["Path_Setting","Config_Setting","Model_Setting","System_Setting","Data_Setting"]

        for i in checklist:
            Setting = config.get(i)
            if Setting is not None:
                print( info() + f"The '{i}' key exists")
            else:
                print( critical() +f"Missing '{i}' key, Please reload and save\n")
                return

        #Seperate different usage
        self.path_setting = config["Path_Setting"]
        self.config_setting = config["Config_Setting"]
        self.model_setting = config["Model_Setting"]
        self.system_setting = config["System_Setting"]
        self.data_setting = config["Data_Setting"]

        print("Finishing Setting/Resetting config....")

