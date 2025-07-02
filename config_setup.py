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
        self.Set_json(self.args.Setting,if_show)

    def update(self,path_setting=None,config_setting=None,model_setting=None,system_setting=None,data_setting=None):
        if path_setting != None:
            self.config["Path_Setting"] = path_setting
        if config_setting != None:
            self.config["Config_Setting"] = config_setting
        if model_setting != None:
            self.config["Model_Setting"] = model_setting
        if system_setting != None:
            self.config["System_Setting"] = system_setting
        if data_setting != None:
            self.config["Data_Setting"] = data_setting
        with open('./Config/Edit.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)

        self.Set_json('./Config/Edit.json',False)
        

    def Set_json(self,json_file,if_show):
        # Load json
        print(f"Loading json => {json_file}...")
        
        self.config = tools.load_json(json_file)
        

        if if_show:
            print(f"Get json:\n {json.dumps(self.config, indent=4, ensure_ascii=False)}")


        #Check exist
        checklist = ["Path_Setting","Config_Setting","Model_Setting","System_Setting","Data_Setting"]

        for i in checklist:
            Setting = self.config.get(i)
            if Setting is not None:
                print( info() + f"The '{i}' key exists")
            else:
                print( critical() +f"Missing '{i}' key, Please reload and save\n")
                return

        #Seperate different usage
        self.path_setting = self.config["Path_Setting"]
        self.config_setting = self.config["Config_Setting"]
        self.model_setting = self.config["Model_Setting"]
        self.system_setting = self.config["System_Setting"]
        self.data_setting = self.config["Data_Setting"]

        print("Finishing Setting/Resetting config....")

