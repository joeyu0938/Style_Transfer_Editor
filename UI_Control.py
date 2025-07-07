import sys
from PyQt6 import QtCore, QtGui, QtWidgets,QtMultimedia, QtMultimediaWidgets
from PyQt6.QtMultimedia import *
from PyQt6.QtGui import QStandardItemModel, QStandardItem,QFileSystemModel
from UI import Ui_MainWindow
from config_setup import Config_setup
from data_edit import Data_Edit
import os
import io
import json
from VLM import VLM
from glob import glob
from Tools.tool import warning,info,critical
import time

class UI_Controller:

    def __init__(self,config:Config_setup):
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        self.config = config
        self.verbose = config.system_setting["verbose"]

        # Set UI infromation
        global folder_contr
        folder_contr = Folder_controller(ui.treeView,config)
        global qmenu_contr
        qmenu_contr = Qmenu_controller(ui,config)
        global graphic_contr
        graphic_contr = Graphic_controller(ui)
        # global System_console_contr
        # System_console_contr = System_Console_controller(ui)
        global Dataedit_contr
        Dataedit_contr = Dataedit_controller(ui,config)
        global config_contr
        config_contr = Config_controller(ui,config)
        global Vlm_contr
        Vlm_contr = VLM_controller(ui,config)
        

        MainWindow.show()
        app.exec()

    def Shutdown():
        print("App Shutting down")
        sys.exit()

class Dataedit_controller(Data_Edit):

    def __init__(self,ui:Ui_MainWindow,config:Config_setup):


        
        super().__init__(config)
        self.Canny_run = ui.Push_1
        self.Lineart_run = ui.Push_2
        self.Depth_run = ui.Push_3
        self.Dilate_run = ui.Push_4
        self.Inpaint_run = ui.Push_5
        self.Dwpose_run = ui.Push_6

        self.ui = ui
        #Set up Canny 
        self.ui.Detect_res_1.setText(str(self.config.data_setting["Canny"]["Detect_res"]))
        self.ui.Image_res_1.setText(str(self.config.data_setting["Canny"]["Image_res"]))
        self.ui.Low_thres_1.setText(str(self.config.data_setting["Canny"]["Low_thres"]))
        self.ui.High_thres_1.setText(str(self.config.data_setting["Canny"]["High_thres"]))
        #Set up Lineart
        self.ui.Detect_res_2.setText(str(self.config.data_setting["Lineart"]["Detect_res"]))
        self.ui.Image_res_2.setText(str(self.config.data_setting["Lineart"]["Image_res"]))
        self.ui.coarse_2.setText(str(self.config.data_setting["Lineart"]["coarse"]))
        #Set up Mask Area Crop
        self.ui.Crop_scale.setText(str(self.config.data_setting["Crop"]["Crop_scale"]))
        self.ui.Crop_width.setText(str(self.config.data_setting["Crop"]["Crop_width"]))
        self.ui.Crop_height.setText(str(self.config.data_setting["Crop"]["Crop_height"]))
        #Set up Depth
        self.ui.Detect_res_3.setText(str(self.config.data_setting["Depth"]["Detect_res"]))
        self.ui.Image_res_3.setText(str(self.config.data_setting["Depth"]["Image_res"]))
        #Set up Dilate Mask
        self.ui.Kernal_3.setText(str(self.config.data_setting["Dilate_Mask"]["Kernal"]))
        self.ui.Iter_3.setText(str(self.config.data_setting["Dilate_Mask"]["Iter"]))
        self.ui.Gaussian_blur_3.setText(str(self.config.data_setting["Dilate_Mask"]["Gaussian"]))
        #Set up Inpaint
        self.ui.radius_5.setText(str(self.config.data_setting["Inpaint"]["Radius"]))
        #Set up Dwpose
        self.ui.Detect_res_6.setText(str(self.config.data_setting["Dwpose"]["Detect_res"]))
        self.ui.Image_res_6.setText(str(self.config.data_setting["Dwpose"]["Image_res"]))
        self.ui.radioButton_hand.setChecked(self.config.data_setting["Dwpose"]["hand"])
        self.ui.radioButton_body.setChecked(self.config.data_setting["Dwpose"]["body"])
        self.ui.radioButton_face.setChecked(self.config.data_setting["Dwpose"]["face"])

        thread_canny = QtCore.QThread()  
        thread_canny.run = self.SendCanny    
        thread_lienart = QtCore.QThread()   
        thread_lienart.run = self.SendLineart     
        thread_depth = QtCore.QThread()   
        thread_depth.run = self.SendDepth  
        thread_dwpose = QtCore.QThread()   
        thread_dwpose.run = self.SendDWpose  
        thread_dilate = QtCore.QThread()   
        thread_dilate.run = self.SendDilate
        thread_inpaint = QtCore.QThread()   
        thread_inpaint.run = self.SendInpaint
        thread_crop = QtCore.QThread()   
        thread_crop.run = self.SendCrop
        self.ui.Push_1.clicked.connect(lambda: thread_canny.start())
        self.ui.Push_2.clicked.connect(lambda: thread_lienart.start())
        self.ui.Push_3.clicked.connect(lambda: thread_depth.start())
        self.ui.Push_4.clicked.connect(lambda: thread_dilate.start())
        self.ui.Push_5.clicked.connect(lambda: thread_inpaint.start())
        self.ui.Push_6.clicked.connect(lambda: thread_dwpose.start())
        self.ui.Push_crop.clicked.connect(lambda: thread_crop.start())
    
    

    def SendCanny(self):
        try:
            images = glob(self.config.path_setting["Image_folder"]+'/*')
            self.config.data_setting["Canny"]["Detect_res"] = int(self.ui.Detect_res_1.text())
            self.config.data_setting["Canny"]["Image_res"] = int(self.ui.Image_res_1.text())
            self.config.data_setting["Canny"]["Low_thres"] = int(self.ui.Low_thres_1.text())
            self.config.data_setting["Canny"]["High_thres"] = int(self.ui.High_thres_1.text())
            self.Canny(images,detect_res=int(self.ui.Detect_res_1.text()),
                    image_res=int(self.ui.Image_res_1.text()),
                    low_thres=int(self.ui.Low_thres_1.text()),
                    high_thres=int(self.ui.High_thres_1.text()),
                    UI_Bar= self.ui.progressBar_1
                    )
            self.config.update(data_setting=self.config.data_setting)
            print(info() + "Finish Canny")
        except:
            print(critical() + " Canny error")
    
    def SendLineart(self):
        try:
            images = glob(self.config.path_setting["Image_folder"]+'/*')
            self.config.data_setting["Lineart"]["Detect_res"] = int(self.ui.Detect_res_2.text())
            self.config.data_setting["Lineart"]["Image_res"] = int(self.ui.Image_res_2.text())
            self.Lineart(images,detect_res=int(self.ui.Detect_res_2.text()),
                    image_res=int(self.ui.Image_res_2.text()),
                    UI_Bar= self.ui.progressBar_2
                    )
            self.config.update(data_setting=self.config.data_setting)
            print(info() + "Finish Lineart")
        except  Exception as e:
            print(critical() + " Lineart error: " + f'{e}' )

    def SendDepth(self):
        try:
            images = glob(self.config.path_setting["Image_folder"]+'/*')
            self.config.data_setting["Depth"]["Detect_res"] = int(self.ui.Detect_res_3.text())
            self.config.data_setting["Depth"]["Image_res"] = int(self.ui.Image_res_3.text())
            self.Depth(images,detect_res=int(self.ui.Detect_res_3.text()),
                    image_res=int(self.ui.Image_res_3.text()),
                    UI_Bar= self.ui.progressBar_3
                    )
            self.config.update(data_setting=self.config.data_setting)
            print(info() + "Finish Depth")
        except:
            print(critical() + " Depth error")
    
    def SendCrop(self):
        # try:
        masks = glob(self.config.path_setting["Mask_folder"]+'/*')
        images = glob(self.config.path_setting["Image_folder"]+'/*')
        self.config.data_setting["Crop"]["Crop_scale"] = float(self.ui.Crop_scale.text())
        self.config.data_setting["Crop"]["Crop_width"] = int(self.ui.Crop_width.text())
        self.config.data_setting["Crop"]["Crop_height"] = int(self.ui.Crop_height.text())
        
        self.Crop(images,masks,
                crop_scale= float(self.ui.Crop_scale.text()),
                crop_width= int(self.ui.Crop_width.text()),
                crop_height= int(self.ui.Crop_height.text()),
                UI_Bar= self.ui.progressBar_crop
                )
        print(info() + "Finish Cropping ")
        print(info() + "Finish Reset Image&MaskFolder (If you hope to use original mask.Please modified config file before running)")
        self.config.path_setting["Image_folder"] = self.config.path_setting["output_folder"] + f'/Tmp/Image'
        self.config.path_setting["Mask_folder"] = self.config.path_setting["output_folder"] + f'/Tmp/Mask'
        self.config.update(data_setting=self.config.data_setting,path_setting=self.config.path_setting)
        config_contr.setPlainText()

        # except  Exception as e:
        #     print(critical() + " Crop error: " + f'{e}')

    def SendDilate(self):
        try:
            masks = glob(self.config.path_setting["Mask_folder"]+'/*')
            self.config.data_setting["Dilate_Mask"]["Kernal"] = int(self.ui.Kernal_3.text())
            self.config.data_setting["Dilate_Mask"]["Iter"] = int(self.ui.Iter_3.text())
            self.config.data_setting["Dilate_Mask"]["Gaussian"] = int(self.ui.Gaussian_blur_3.text())
            
            self.Dilate(masks,kernal=int(self.ui.Kernal_3.text()),
                    iter=int(self.ui.Iter_3.text()),
                    gaussian=int(self.ui.Gaussian_blur_3.text()),
                    UI_Bar= self.ui.progressBar_4
                    )
            print(info() + "Finish Dilate")
            print(info() + "Finish Reset Mask Folder (If you hope to use original mask.Please modified config file before running)")
            self.config.path_setting["Mask_folder"] = self.config.path_setting["output_folder"] + f'/Tmp/Mask'
            self.config.update(data_setting=self.config.data_setting,path_setting=self.config.path_setting)
            config_contr.setPlainText()

        except  Exception as e:
            print(critical() + " Dilate error: " + f'{e}')

    def SendInpaint(self):
        try:
            images = glob(self.config.path_setting["Image_folder"]+'/*')
            masks = glob(self.config.path_setting["Mask_folder"]+'/*')
            self.config.data_setting["Inpaint"]["Radius"] = int(self.ui.radius_5.text())
            self.Inpaint(images,masks,radius=int(self.ui.radius_5.text()),
                    UI_Bar= self.ui.progressBar_5
                    )
            print(info() + "Finish Inpaint")
            print(info() + "Finish Reset Image Folder (If you hope to use original mask.Please modified config file before running)")
            self.config.path_setting["Image_folder"] = self.config.path_setting["output_folder"] + f'/Tmp/Image'
            self.config.update(data_setting=self.config.data_setting,path_setting=self.config.path_setting)
            config_contr.setPlainText()
        except  Exception as e:
            print(critical() + " Inpaint error: " + f'{e}')

    def SendDWpose(self):
        try:
            images = glob(self.config.path_setting["Image_folder"]+'/*')
            self.config.data_setting["Dwpose"]["Detect_res"] = int(self.ui.Detect_res_6.text())
            self.config.data_setting["Dwpose"]["Image_res"] = int(self.ui.Image_res_6.text())
            self.config.data_setting["Dwpose"]["hand"] = self.ui.radioButton_hand.isChecked()
            self.config.data_setting["Dwpose"]["body"] = self.ui.radioButton_body.isChecked()
            self.config.data_setting["Dwpose"]["face"] = self.ui.radioButton_face.isChecked()
            self.Dwpose(images,detect_res=int(self.ui.Detect_res_6.text()),
                    image_res=int(self.ui.Image_res_6.text()),
                    hand=self.ui.radioButton_hand.isChecked(),
                    body=self.ui.radioButton_body.isChecked(),
                    face=self.ui.radioButton_face.isChecked(),
                    UI_Bar= self.ui.progressBar_6
                    )
            self.config.update(data_setting=self.config.data_setting)
            print(info() + "Finish Dwpose")
        except:
            print(critical() + " Dwpose error")


class Config_controller():
    

    def __init__(self,ui:Ui_MainWindow,config:Config_setup):
        
        super().__init__()
        self.config= config 
        self.Path_browser = ui.textBrowser_3
        self.Config_browser= ui.textBrowser
        self.Model_browser = ui.textBrowser_2
        self.System_browser = ui.textBrowser_System_Conifg
        self.Path_enable = ui.Path_enable
        self.Config_enable = ui.Config_enable
        self.Model_enable = ui.Model_enable
        self.System_enable = ui.System_enable
        self.lock = True


        self.Refresh = ui.pushButton_16

        self.Path_browser.setPlainText(json.dumps(config.path_setting,sort_keys=True, indent=2))
        self.Config_browser.setPlainText(json.dumps(config.config_setting,sort_keys=True, indent=2))
        self.Model_browser.setPlainText(json.dumps(config.model_setting,sort_keys=True, indent=2))
        self.System_browser.setPlainText(json.dumps(config.system_setting,sort_keys=True, indent=2))
        
        

        self.Refresh.clicked.connect(self.Refresh_configuration)
        self.Path_enable.checkStateChanged.connect(lambda: self.Setlock(self.Path_browser))
        self.Config_enable.checkStateChanged.connect(lambda: self.Setlock(self.Config_browser))
        self.Model_enable.checkStateChanged.connect(lambda: self.Setlock(self.Model_browser))
        self.System_enable.checkStateChanged.connect(lambda: self.Setlock(self.System_browser))

        self.Path_browser.setDisabled(self.lock)
        self.Config_browser.setDisabled(self.lock)
        self.Model_browser.setDisabled(self.lock)
        self.System_browser.setDisabled(self.lock)

    def Refresh_configuration(self):
        path_setting = json.loads(self.Path_browser.toPlainText())
        config_setting = json.loads(self.Config_browser.toPlainText())
        model_setting = json.loads(self.Model_browser.toPlainText())
        system_setting = json.loads(self.System_browser.toPlainText())
        self.config.update(path_setting=path_setting,config_setting=config_setting,model_setting=model_setting,system_setting=system_setting)
        self.setPlainText()

    def setPlainText(self):
        self.Path_browser.setPlainText(json.dumps(self.config.path_setting,sort_keys=True, indent=2))
        self.Config_browser.setPlainText(json.dumps(self.config.config_setting,sort_keys=True, indent=2))
        self.Model_browser.setPlainText(json.dumps(self.config.model_setting,sort_keys=True, indent=2))
        self.System_browser.setPlainText(json.dumps(self.config.system_setting,sort_keys=True, indent=2))

    def Setlock(self,item:QtWidgets.QPlainTextEdit):
        if item.isEnabled() == False:
            item.setEnabled(True)
        else:
            item.setEnabled(False)

class Folder_controller:

    def __init__(self,tree:QtWidgets.QTreeView,config:Config_setup):
        self.treeview = tree
        self.treeview.doubleClicked.connect(self.on_item_double_clicked)
        self.verbose = config.system_setting["verbose"]
        self.config = config
        self.folder_viewer = QFileSystemModel()
        self.folder_viewer.setRootPath(self.config.system_setting["Viewer_load_path"])
        self.treeview.setModel(self.folder_viewer)
        self.update()

    def on_item_double_clicked(self,index: QtCore.QModelIndex):
        index = self.treeview.currentIndex()
        model = index.model()
        path = model.fileInfo(index).absoluteFilePath()
        graphic_contr.check_task(path)

    def update(self):
        #set folder viwer
        print(info() + f"Updating Folder viewer {self.config.system_setting["Viewer_load_path"]}")
        self.treeview.setRootIndex(self.folder_viewer.index(self.config.system_setting["Viewer_load_path"]))
        
class Qmenu_controller:
    
    def __init__(self,ui:Ui_MainWindow,config:Config_setup):
        
        ui.actionOpen_folder.triggered.connect(self.open_folder)
        ui.actionOpen_file.triggered.connect(self.open_file)
        ui.actionExit.triggered.connect(self.shutdown)
        self.config = config

    def open_folder(self):
        filePath = QtWidgets.QFileDialog.getExistingDirectory()  # 選擇檔案對話視窗
        print(info() + f'Opening Folder" {filePath}')
        self.config.system_setting["Viewer_load_path"] = filePath
        folder_contr.update()

    def open_file(self):
        filePath,type = QtWidgets.QFileDialog.getOpenFileName()  # 選擇檔案對話視窗
        print(info() + f'Opening Folder" {filePath}')
        self.config.system_setting["Viewer_load_path"] = filePath
        folder_contr.update()

    def shutdown(self):
        UI_Controller.Shutdown()
    
    def update(self):
        folder_contr.update()

class VLM_controller(VLM):
    def __init__(self,ui:Ui_MainWindow,config:Config_setup):
        super().__init__(config)
        self.ui = ui 
        self.config = config
        self.ui.Image_path.setText(config.config_setting["reference_img"])
        self.ui.Prompt_input.setText("Describe image in english prompt 1 and prompt2 with less than 77 words ")

        self.ui.Recursive_run.clicked.connect(self.Multi_run)
        self.ui.Send_prompt.clicked.connect(self.Single_run)
        self.ui.ClearVLM.clicked.connect(self.unload_model)

    def Multi_run(self):
        self.load_model()
        self.images = glob(self.ui.Image_path.text()+'/*.jpg') + glob(self.ui.Image_path.text()+'/*.png')
        self.images  = sorted(self.images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        response_list = list()
        print(info()+"Running VLM recursively")
        for i in self.images:
            response_list.append(self.run_vlm(self.ui.Prompt_input.text(),i)[0])
        self.ui.Response.setText(f"Running multiple images recursively to prompt1 and promp2: \n {response_list}")

    def Single_run(self):
        self.load_model()
        print(info()+"Running VLM singly")
        response = self.run_vlm(self.ui.Prompt_input.text(),self.ui.Image_path.text())[0]
        self.ui.Response.setText(response)
        



class Graphic_controller:

    def __init__(self,ui:Ui_MainWindow):
        self.Current_frame = 0
        self.image_list = list()
        self.grview = ui.graphicsView
        self.slider = ui.horizontalSlider
        self.error = False
        
        # Setup Button control
        self.play_button = ui.Play_pause
        self.next_button = ui.next
        self.prev_button = ui.prev
        self.spin_button = ui.spinBox
        self.play_pause_st = 1
        self.play_button.clicked.connect(self.play)
        self.next_button.clicked.connect(self.next)
        self.prev_button.clicked.connect(self.prev)
        self.spin_button.valueChanged.connect(self.spin_change)

        #Set up Current directory
        self.Current_path = ui.Show_Directory


        self.grview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.task = "Folder" #Set folder or task for render
        self.scene_img = QtWidgets.QGraphicsScene()
        self.scene_vid = QtWidgets.QGraphicsScene()

        self.slider.setRange(0, 100)
        self.slider.setValue(0)


    def check_task(self,path:str):
        if os.path.isdir(path): 
            self.Current_frame = 0
            self.slider.setRange(0, 100)
            self.slider.setValue(0)
            self.spin_button.setValue(0)
            self.task = "Folder"
            self.images = glob(path+'/*.jpg') + glob(path+'/*.png')
            if len(self.images) == 0:
                print(critical() + f"No images in folder: {path}")
                self.error = True
                return
            self.slider.setMaximum(len(self.images)-1)
            self.slider.sliderMoved.connect(self.load_folder)
            self.Current_path.setText(f"Current Folder Path: {path}")
            print(info() + f"Loading images in folder: {path}")
            self.error = False
            self.load_folder()
        elif os.path.isfile(path):
            self.Current_frame = 0
            self.slider.setRange(0, 100)
            self.slider.setValue(0)
            self.spin_button.setValue(0)
            # Check if it's an image or a video
            _, file_extension = os.path.splitext(path)
            file_extension = file_extension.lower()

            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'] # Add more as needed
            image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'] # Add more as needed

            if file_extension in video_extensions:
                self.task = "Video"
                self.Current_path.setText(f"Current Video Path: {path}")
                self.load_video(path)
                print(info() + f"Loading video file: {path}")
                self.error = False

            elif file_extension in image_extensions:
                self.task = "File"
                self.Current_path.setText(f"Current file Path: {path}")
                self.load_image(path)
                print(info() + f"Loading image file: {path}")
                self.error = False

            else:
                print(critical() + f"Unknown path: {path}")
                self.error = True

    def load_video(self,path):

        self._videoitem = QtMultimediaWidgets.QGraphicsVideoItem()
        self.scene_vid.clear()
        self.scene_vid.addItem(self._videoitem)
        
        self._player = QtMultimedia.QMediaPlayer()
        self._player.setVideoOutput(self._videoitem)
        self._player.setSource(QtCore.QUrl.fromLocalFile(path))
        self._player.positionChanged.connect(self.position_change)
        self._player.durationChanged.connect(lambda: self.slider.setMaximum(self._player.duration())) 
        self.slider.sliderMoved.connect(lambda: self._player.setPosition(self.slider.value()))
        
        self._videoitem.setSize(QtCore.QSizeF(self.grview.size()))
        self.scene_vid.setSceneRect(self._videoitem.boundingRect())
        self.grview.setScene(self.scene_vid)
        self.grview.show()

    def position_change(self):
        self.slider.setValue(self._player.position())
        self.Current_frame = self._player.position()
        self.spin_button.setValue(int(self._player.position()/1000))

    def load_image(self,path):
        # Ensure video widget is hidden when displaying an image
        self.scene_img.clear()
        self.grview.setScene(self.scene_img)
        img = QtGui.QPixmap(path)
        img = img.scaled(self.grview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.scene_img.setSceneRect(0, 0, img.width(), img.height())
        self.scene_img.addPixmap(img)
        self.grview.show()
    
    def load_folder(self):
        # Ensure video widget is hidden when displaying an image
        if self.task == "Folder":
            self.Current_frame = self.slider.value()
            self.spin_button.setValue(self.Current_frame)
            self.scene_img.clear()
            self.grview.setScene(self.scene_img)
            if self.Current_frame >= len(self.images):
                self.Current_frame = len(self.images)-1
            elif self.Current_frame <= 0:
                self.Current_frame = 0
            img = QtGui.QPixmap(self.images[self.Current_frame])
            img = img.scaled(self.grview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            self.scene_img.setSceneRect(0, 0, img.width(), img.height())
            self.scene_img.addPixmap(img)
            self.grview.show()

    def play(self):
        if self.task == "Video":
            if self.play_pause_st == 1:
                self._player.play()
                self.play_pause_st = 0
            else:
                self._player.pause()
                self.play_pause_st = 1

    def next(self):
        if self.task == "Video" and not self.error:
            self.Current_frame += 250
            self._player.setPosition(self.Current_frame)
        elif self.task == "Folder" and not self.error:
            self.Current_frame += 1
            self.slider.setSliderPosition(self.Current_frame)
            self.load_folder()

    def prev(self):
        if self.task == "Video" and not self.error:
            self.Current_frame -= 250
            self._player.setPosition(self.Current_frame)
        elif self.task == "Folder" and not self.error:
            self.Current_frame -= 1
            self.slider.setSliderPosition(self.Current_frame)
            self.load_folder()

    def spin_change(self):
        if self.task == "Video" and self._player.playbackState() == QMediaPlayer.PlaybackState.PausedState and not self.error:
            self._player.setPosition(int(self.spin_button.value()*1000))
        elif self.task == "Folder" and not self.error:
            self.Current_frame = self.spin_button.value()
            self.slider.setSliderPosition(self.Current_frame)
            self.load_folder()