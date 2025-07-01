import sys
from PyQt6 import QtCore, QtGui, QtWidgets,QtMultimedia, QtMultimediaWidgets
from PyQt6.QtMultimedia import *
from PyQt6.QtGui import QStandardItemModel, QStandardItem,QFileSystemModel
from UI import Ui_MainWindow
from config_setup import Config_setup
import os
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


        MainWindow.show()
        app.exec()

    def Shutdown():
        print("App Shutting down")
        sys.exit()

class Folder_controller:

    def __init__(self,tree:QtWidgets.QTreeView,config:Config_setup):
        self.treeview = tree
        self.treeview.doubleClicked.connect(self.on_item_double_clicked)
        self.verbose = config.system_setting["verbose"]
        self.config = config
        self.folder_viewer = QFileSystemModel()
        self.folder_viewer.setRootPath(self.config.system_setting["Load_path"])
        self.treeview.setModel(self.folder_viewer)
        self.update()

    def on_item_double_clicked(self,index: QtCore.QModelIndex):
        index = self.treeview.currentIndex()
        model = index.model()
        path = model.fileInfo(index).absoluteFilePath()
        graphic_contr.check_task(path)

    def update(self):
        #set folder viwer
        print(info() + f"Updating Folder viewer {self.config.system_setting["Load_path"]}")
        self.treeview.setRootIndex(self.folder_viewer.index(self.config.system_setting["Load_path"]))
        
class Qmenu_controller:
    
    def __init__(self,ui:Ui_MainWindow,config:Config_setup):
        
        ui.actionOpen_folder.triggered.connect(self.open_folder)
        ui.actionOpen_file.triggered.connect(self.open_file)
        ui.actionExit.triggered.connect(self.shutdown)
        self.config = config

    def open_folder(self):
        filePath = QtWidgets.QFileDialog.getExistingDirectory()  # 選擇檔案對話視窗
        print(info() + f'Opening Folder" {filePath}')
        self.config.system_setting["Load_path"] = filePath
        folder_contr.update()

    def open_file(self):
        filePath,type = QtWidgets.QFileDialog.getOpenFileName()  # 選擇檔案對話視窗
        print(info() + f'Opening Folder" {filePath}')
        self.config.system_setting["Load_path"] = filePath
        folder_contr.update()

    def shutdown(self):
        UI_Controller.Shutdown()
    
    def update(self):
        folder_contr.update()

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
            self.slider.valueChanged.connect(self.folder_slider)
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
        self.grview.setScene(self.scene_vid)
        self._player = QtMultimedia.QMediaPlayer()
        self._player.setVideoOutput(self._videoitem)
        self._player.setSource(QtCore.QUrl.fromLocalFile(path))
        self.grview.fitInView(self._videoitem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self._player.positionChanged.connect(self.position_change)
        self._player.durationChanged.connect(lambda: self.slider.setMaximum(self._player.duration())) 
        self.slider.sliderMoved.connect(lambda: self._player.setPosition(self.slider.value()))
        self.grview.show()
    
    def position_change(self):
        self.slider.setValue(self._player.position())
        self.Current_frame = self._player.position()
        self.spin_button.setValue(int(self._player.position()/1000))

    def load_image(self,path):
        # Ensure video widget is hidden when displaying an image
        self.grview.setScene(self.scene_img)
        img = QtGui.QPixmap(path)
        img = img.scaled(self.grview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.scene_img.clear()
        self.scene_img.setSceneRect(0, 0, img.width(), img.height())
        self.scene_img.addPixmap(img)
        self.grview.show()
    
    def load_folder(self):
        # Ensure video widget is hidden when displaying an image
        self.grview.setScene(self.scene_img)
        if self.Current_frame >= len(self.images):
            self.Current_frame = len(self.images)-1
        elif self.Current_frame <= 0:
            self.Current_frame = 0
        img = QtGui.QPixmap(self.images[self.Current_frame])
        img = img.scaled(self.grview.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.scene_img.clear()
        self.scene_img.setSceneRect(0, 0, img.width(), img.height())
        self.scene_img.addPixmap(img)
        self.slider.setValue(self.Current_frame)
        self.grview.show()
        
    def folder_slider(self):
        self.Current_frame = self.slider.value()
        self.spin_button.setValue(self.Current_frame)
        self.load_folder()

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
            self.load_folder()

    def prev(self):
        if self.task == "Video" and not self.error:
            self.Current_frame -= 250
            self._player.setPosition(self.Current_frame)
        elif self.task == "Folder" and not self.error:
            self.Current_frame -= 1
            self.load_folder()

    def spin_change(self):
        if self.task == "Video" and self._player.playbackState() == QMediaPlayer.PlaybackState.PausedState and not self.error:
            self._player.setPosition(int(self.spin_button.value()*1000))
        elif self.task == "Folder" and not self.error:
            self.Current_frame = self.spin_button.value()
            self.load_folder()