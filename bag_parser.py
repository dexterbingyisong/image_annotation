import os
envpath = '/home/nio/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
#coding=utf-8
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import threading
import cv2

class BagParser(threading.Thread):

    def __init__(self, bag_file):
        super(BagParser, self).__init__()
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()

        self.bag_file = bag_file
        self.all_frames = []
        self.frame_names = []

    def __del__(self):
        self._stop_event.set()

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

    def run(self):
        print("BagParser open file ",self.bag_file)
        image_list = []
        filenames = os.listdir(self.bag_file)
        filenames.sort(key=lambda x: x[:-4])
        for file in filenames:

            self.all_frames.append(os.path.join(self.bag_file,file)) 
            self.frame_names.append(file)

        self.stop()

    def get_all_frames(self):
        return self.all_frames

    def get_frame_names(self):
        return self.frame_names
