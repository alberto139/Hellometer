import cv2
import numpy as np  
import utils
import time
import random


class Object():
    def __init__(self, frame, box, cls):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        self.id = 0 # all objects are initialy id 0, until matched
        self.box = [int(box[0] * frame_height), int(box[1] * frame_width), int(box[2] * frame_height), int(box[3] * frame_width)]
        self.width = int(box[3] * frame_width) - int(box[1] * frame_width)
        self.height = int(box[2] * frame_height) -  int(box[0] * frame_height)
        (self.ymin, self.xmin, self.ymax, self.xmax) = self.box
        self.frame = frame
        self.subimg = frame[self.ymin:self.ymax, self.xmin:self.xmax]
        self.x = self.xmin +(self.xmax - self.xmin) // 2
        self.y = (self.ymin +(self.ymax - self.ymin) // 2)
        self.y = int(self.y) # increase y to compensate for viewing angle
        self.center = (self.x, self.y)
        self.prev_centers = [self.center]
        self.prev_times = [time.time()]
        self.match = False
        self.missing = 0
        self.cls = cls
        self.cls_string = utils.cls2label(self.cls)
        self.cross_time = 0.0
        self.cross_times = []
        self.time_string = utils.epoch_to_human(self.cross_time)
        self.crossed = False
        self.aoi_ids = [] # list of aois crossed
        self.start_time = time.time()
        self.last_detected = self.start_time
        self.first_cross = True
        self.direction = ""
        self.directions = []
        self.speed = 0
        self.conf = 0
        self.active = False
        self.color = (random.randrange(255), random.randrange(255), random.randrange(255))
        self.waittime = 0 
        self.type = ''
