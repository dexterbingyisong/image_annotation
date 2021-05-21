import os
envpath = '/home/nio/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget, QComboBox, QTextEdit
from PyQt5.QtGui import QPainter, QPixmap, QImage
import sys
import cv2
import numpy as np
import time

from bag_parser import BagParser

import pyqtgraph as pg
import json

class MyPopWidget(QWidget):
    def __init__(self,roi):
        QWidget.__init__(self)
        self.setWindowTitle("Attribute")
        self.resize(300,270)

        self.roi = roi

        self.layout = QGridLayout()

        self.label_ID = QLabel("ID:")
        self.label_Label = QLabel("Label:")
        self.label_Status = QLabel("Status:")

        self.comb_Label = QComboBox()
        self.comb_Status = QComboBox()

        self.spinbox_ID = QSpinBox()
        self.bt_cancel = QPushButton("Cancel")
        self.bt_OK = QPushButton("OK")

        self.bt_cancel.clicked.connect(self.press_cancel)
        self.bt_OK.clicked.connect(self.press_OK)

        labels = self.roi.class_status_dict.keys()
        for label in labels:
            self.comb_Label.addItem(label)
        
        id = self.roi.attrs["ID"]
        self.spinbox_ID.setValue(id)

        label = self.roi.attrs["Label"]
        self.comb_Label.setCurrentText(label)

        avaliable_status = self.roi.class_status_dict[label]
        for status in avaliable_status:
            self.comb_Status.addItem(status)
        self.comb_Status.setCurrentIndex(0)

        self.comb_Label.currentIndexChanged.connect(self.update_status)
        self.layout.addWidget(self.label_ID,0,0)
        self.layout.addWidget(self.spinbox_ID,0,1)
        self.layout.addWidget(self.label_Label,1,0)
        self.layout.addWidget(self.comb_Label,1,1)
        
        self.layout.addWidget(self.label_Status,2,0)
        self.layout.addWidget(self.comb_Status,2,1)
        self.layout.addWidget(self.bt_cancel,3,0)
        self.layout.addWidget(self.bt_OK,3,1)

        self.setLayout(self.layout)

    def press_OK(self):
        id = self.spinbox_ID.value()
        label = self.comb_Label.currentText()
        status = self.comb_Status.currentText()
        
        self.roi.attrs["ID"] = id
        self.roi.attrs["Label"] = label
        self.roi.attrs["Status"] = status
        self.close()

    def press_cancel(self):
        self.close()

    def update_status(self):
        label = self.comb_Label.currentText()
        avaliable_status = self.roi.class_status_dict[label]
        self.comb_Status.clear()
        for status in avaliable_status:
            self.comb_Status.addItem(status)
        self.comb_Status.setCurrentIndex(0)

class MyPolyLineROI(pg.PolyLineROI):
    class_status_dict = {   "Parking-slot":["Occupied","Self-occupied","Empty"],
                            "Wheel-block":[""],
                            "Obstacle":[""],
                            "Pillar":[""],
                            "Circular-pillar":[""],
                            "Ground-sign":["forward","right","forward-left","left","turn-around","forward-right","left-right","forward-left-right","unknow"],
                            "Slot-number":[""]
                        }
    def __init__(self, positions, closed=False, pos=None, **args):
        super().__init__(positions, closed=closed, pos=pos, **args)
        self.attrs = {"ID":int(0),"Label":"Parking-slot","Segmentation":[],"Status":""}
        

    def getInfo(self):
        points = []
        for h in self.handles:
            points.append([h['pos'].x(),h['pos'].y()])
        key_points = {"Key_points":points}
        self.attrs.update(key_points)
        return self.attrs

class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # self.image_frame = QtWidgets.QLabel(self)
        self.image_frame = pg.PlotWidget()
        self.init_image_frame()
        self.v_line = pg.InfiniteLine(angle=90,movable=False)
        self.h_line = pg.InfiniteLine(angle=0,movable=False)
        self.pos_label = pg.TextItem()
        self.image_frame.addItem(self.v_line)
        self.image_frame.addItem(self.h_line)
        self.image_frame.addItem(self.pos_label)

        self.mouse_moved_proxy = pg.SignalProxy(self.image_frame.scene().sigMouseMoved, rateLimit=60,slot=self.mouse_moved)
        self.mouse_clicked_proxy = pg.SignalProxy(self.image_frame.scene().sigMouseClicked , rateLimit=60,slot=self.mouse_clicked)
        
        self.curr_state = "NonePoint_NonePoly"
        self.mouse_point = []
        self.polygons = []
        self.polygon_points = []

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_frame)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.bt_open = QtWidgets.QPushButton('Open Bag')
        self.bt_open.clicked.connect(self.open_bag)
        self.button_layout.addWidget(self.bt_open)

        self.bt_prev = QtWidgets.QPushButton('Prev')
        self.bt_prev.clicked.connect(self.prev_image)
        self.button_layout.addWidget(self.bt_prev)

        self.bt_next = QtWidgets.QPushButton('Next')
        self.bt_next.clicked.connect(self.next_image)
        self.button_layout.addWidget(self.bt_next)

        self.bt_save = QtWidgets.QPushButton('Save')
        self.bt_save.clicked.connect(self.save_json)
        self.button_layout.addWidget(self.bt_save)

        self.progress_label = QtWidgets.QLabel()
        self.progress_label.setText("Progress: 0/0")
        self.button_layout.addWidget(self.progress_label)

        self.save_cnt_label = QtWidgets.QLabel()
        self.save_cnt_label.setText("Saved: 0")
        self.button_layout.addWidget(self.save_cnt_label)

        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

        self.all_frames = []
        self.bag_thd = None
        self.save_folder = None
        self.save_cnt = 0


    def init_image_frame(self):
        empty_image = np.zeros((640,480,3))
        self.image = QtGui.QImage(empty_image.data, 640, 480,
                                  QtGui.QImage.Format_RGB888).rgbSwapped()
        self.q_img = pg.QtGui.QGraphicsPixmapItem(pg.QtGui.QPixmap.fromImage(self.image))
        self.image_frame.addItem(self.q_img)
        self.image_frame.setAspectLocked()
        self.image_frame.getPlotItem().hideAxis("left")
        self.image_frame.getPlotItem().hideAxis("bottom")
        self.image_frame.setMenuEnabled(False)

    def open_bag(self):
        filename = QFileDialog.getExistingDirectory(self,"Open file","./")
        print(filename)
        bag_file = filename

        self.save_cnt = 0
        self.save_cnt_label.setText("Saved: 0")
        self.frame_cnt = 0
        self.update_progress()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.preview_timer)
        self.timer.start(100)

        self.bag_thd = BagParser(bag_file)
        self.bag_thd.setDaemon(True)
        self.bag_thd.start()



    def preview_timer(self):
        if self.bag_thd is not None:
            self.all_frames = self.bag_thd.get_all_frames()
            self.frame_names = self.bag_thd.get_frame_names()

        if len(self.all_frames) == 0:
            return

        frame_path = self.all_frames[-1]
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        self.image = QtGui.QImage(frame.data,frame.shape[1],frame.shape[0], 
                                  QtGui.QImage.Format_RGB888).rgbSwapped()
        self.q_img.setPixmap(pg.QtGui.QPixmap.fromImage(self.image))


        if self.bag_thd.is_stopped():
            self.timer.stop()
            self.frame_cnt = 0
            self.update_image(self.frame_cnt)
            self.update_progress()


    def prev_image(self):
        self.frame_cnt -= 1
        self.frame_cnt = max(0, self.frame_cnt)
        self.update_image(self.frame_cnt)
        self.update_progress()
        for poly in self.polygons:
            self.image_frame.removeItem(poly)
        self.polygons.clear()
        self.polygon_points.clear()
        self.curr_state = "NonePoint_NonePoly"

    def next_image(self):
        self.frame_cnt += 1
        self.frame_cnt = min(len(self.all_frames) - 1, self.frame_cnt)
        self.update_image(self.frame_cnt)
        self.update_progress()
        for poly in self.polygons:
            self.image_frame.removeItem(poly)
        self.polygons.clear()
        self.polygon_points.clear()
        self.curr_state = "NonePoint_NonePoly"

    def update_image(self, frame_cnt):
        frame_path = self.all_frames[frame_cnt]
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        self.image = QtGui.QImage(frame.data,frame.shape[1],frame.shape[0],   
                                  QtGui.QImage.Format_RGB888).rgbSwapped()
        self.q_img.setPixmap(pg.QtGui.QPixmap.fromImage(self.image))
        self.image_frame.setRange(xRange=[0, frame.shape[1]],
             yRange=[0,frame.shape[0]], padding=0)

    def update_progress(self):
        progress = "Progress: {0}/{1}".format(self.frame_cnt, len(self.all_frames))
        self.progress_label.setText(progress)

    def save_image(self):
        if self.save_folder is None:
            self.save_folder = QFileDialog.getExistingDirectory()
        frame_path = self.all_frames[self.frame_cnt]
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        print(self.frame_cnt)
        print(self.frame_names)
        frame_name = self.frame_names[self.frame_cnt]
        folder_name = frame_name.split("/")[0]
        file_name = frame_name.split("/")[-1]

        curr_save_path = self.save_folder 
        if not os.path.exists(curr_save_path):
            os.mkdir(curr_save_path)

        curr_file = file_name 
        full_file = os.path.join(curr_save_path,curr_file)
        print("save to", full_file)
        cv2.imwrite(full_file, frame) 

        self.save_cnt += 1
        self.save_cnt_label.setText("Saved: "+str(self.save_cnt))

    def save_json(self):
        if self.save_folder is None:
            self.save_folder = QFileDialog.getExistingDirectory()
        frame_path = self.all_frames[self.frame_cnt]
        print(self.frame_cnt)
        print(self.frame_names)
        frame_name = self.frame_names[self.frame_cnt]
        file_name = frame_name.split("/")[-1]
        file_name = file_name.split(".")
        file_name.pop()
        file_name = ".".join(file_name)


        curr_save_path = self.save_folder 
        if not os.path.exists(curr_save_path):
            os.mkdir(curr_save_path)

        curr_file = file_name + ".json"
        full_file = os.path.join(curr_save_path,curr_file)
        print("save to", full_file)

        res = []
        for poly in self.polygons:
            res.append(poly.getInfo())
        
        with open(full_file,"w") as f:
            json.dump(res,f,indent=4)

        self.save_cnt += 1
        self.save_cnt_label.setText("Saved: "+str(self.save_cnt))

    def combine_image(self, image_list):

        img_show_list = []

        for i in range( len(image_list) ):
            img_show_list.append( cv2.resize(image_list[i], (480,320)) )

        if len(image_list)==4:
            combine1 = np.concatenate((img_show_list[1], img_show_list[0]), axis=1)
            combine2 = np.concatenate((img_show_list[2], img_show_list[3]), axis=1)
            combine = np.concatenate((combine1, combine2), axis=0)
            return combine

        elif len(image_list) == 5:
            combine1 = np.concatenate((img_show_list[1], img_show_list[0]), axis=1)
            combine2 = np.concatenate((img_show_list[2], img_show_list[3]), axis=1)
            combine1_2 = np.concatenate((combine1, combine2), axis=0)

            empty_image = np.zeros(img_show_list[4].shape).astype(np.uint8)
            combine3 = np.concatenate((img_show_list[4], empty_image), axis=0)
            combine = np.concatenate((combine1_2, combine3), axis=1)

            return combine


    def mouse_moved(self,evt):
        pos = evt[0] 
        if self.image_frame.sceneBoundingRect().contains(pos):
            mousePoint = self.image_frame.plotItem.vb.mapSceneToView(pos)
            self.pos_label.setHtml(
                "(<span style='font-size: 12pt'>%0.3f, %0.3f</span>)" % (mousePoint.x(), mousePoint.y()))
            self.pos_label.setPos(mousePoint.x(), mousePoint.y())
            self.v_line.setPos(mousePoint.x())
            self.h_line.setPos(mousePoint.y())

    def NonePoint_NonePoly(self,button,doubled):
        if(doubled == 0):
            if(button == 1):

                self.polygon_points.append(self.mouse_point)
                self.polygons.append(MyPolyLineROI(self.polygon_points,True))
                self.polygons[-1].sigClicked.connect(self.roi_clicked)
                self.polygons[-1].setAcceptedMouseButtons(QtCore.Qt.MidButton)
                self.image_frame.addItem(self.polygons[-1])
                self.curr_state = "HavePoint_NonePoly"
        

    def HavePoint_NonePoly(self,button,doubled):
        if(button == 1):
            if(doubled == 0):
                self.polygon_points.append(self.mouse_point)
                self.polygons[-1].setPoints(self.polygon_points,True)
                self.curr_state = "HavePoint_NonePoly"
            else:
                self.polygon_points.pop()
                self.polygons[-1].setPoints(self.polygon_points,True)
                self.polygon_points.clear()
                self.curr_state = "NonePoint_HavePoly"
        elif(button == 2):
            self.polygon_points.pop()
            self.polygons[-1].setPoints(self.polygon_points,True)
            if(len(self.polygon_points)>0):
                self.curr_state = "HavePoint_NonePoly"
            else:
                polygon = self.polygons.pop()
                self.image_frame.removeItem(polygon)
                self.curr_state = "NonePoint_NonePoly"


    def NonePoint_HavePoly(self,button,doubled):
        if(button == 1):
            self.polygon_points.append(self.mouse_point)
            self.polygons.append(MyPolyLineROI(self.polygon_points,True))
            self.polygons[-1].sigClicked.connect(self.roi_clicked)
            self.polygons[-1].setAcceptedMouseButtons(QtCore.Qt.MidButton)
            self.image_frame.addItem(self.polygons[-1])
            self.curr_state = "HavePoint_HavePoly"
        elif(button == 2):
            polygon = self.polygons.pop()
            self.image_frame.removeItem(polygon)
            if(len(self.polygons)>0):
                self.curr_state = "NonePoint_HavePoly"
            else:
                self.curr_state = "NonePoint_NonePoly"
        
    
    def HavePoint_HavePoly(self,button,doubled):
            if(button == 1):
                if(doubled == 0):
                    self.polygon_points.append(self.mouse_point)
                    self.polygons[-1].setPoints(self.polygon_points,True)
                    self.curr_state = "HavePoint_HavePoly"
                else:
                    self.polygon_points.pop()
                    self.polygons[-1].setPoints(self.polygon_points,True)
                    self.polygon_points.clear()
                    self.curr_state = "NonePoint_HavePoly"
            elif(button == 2):
                self.polygon_points.pop()
                self.polygons[-1].setPoints(self.polygon_points,True)
                if(len(self.polygon_points)>0):
                    self.curr_state = "HavePoint_HavePoly"
                else:
                    polygon = self.polygons.pop()
                    self.image_frame.removeItem(polygon)
                    self.curr_state = "NonePoint_HavePoly"
        
    def polygonStateManager(self,button,doubled):
        cur_state = self.curr_state
        if(self.curr_state == "NonePoint_NonePoly"):
            self.NonePoint_NonePoly(button,doubled)
        elif(self.curr_state == "HavePoint_NonePoly"):
            self.HavePoint_NonePoly(button,doubled)
        elif(self.curr_state == "NonePoint_HavePoly"):
            self.NonePoint_HavePoly(button,doubled)
        elif(self.curr_state == "HavePoint_HavePoly"):
            self.HavePoint_HavePoly(button,doubled)
        else:   
            return
        next_state = self.curr_state
        print(cur_state,"->",next_state)

    def mouse_clicked(self,evt):
        event = evt[0]
        pos = event.scenePos()
        mousePoint = self.image_frame.plotItem.vb.mapSceneToView(pos)
        self.mouse_point = [mousePoint.x(),mousePoint.y()]
        print(event.double())
        self.polygonStateManager(event.button(),event.double())

    def roi_clicked(self,roi,ev):
        print("clicked roi")
        self.pop = MyPopWidget(roi)
        self.pop.show()
