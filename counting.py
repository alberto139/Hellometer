import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import csv, time
import pandas as pd





def class_count_area(aois, detections, df, upload, frame_time):

    # iterate over all areas and all objects
    # if an object is seen in an area for the first time, keep track of it
    # if an object previously seen in an area is no longer in the area, record its data in dataframe
    for aoi in aois:
        aoi.active = False
        for obj in detections:
            #obj.active = False


            if len(obj.prev_centers) > 2:
                current_center= obj.prev_centers[-1]
                prev_center = obj.prev_centers[-2]

                point = Point(prev_center[0], prev_center[1])

                                        
                # First time seeing object in area
                if aoi.polygon.contains(point) and (not obj.id in aoi.objects_crossed):
                    aoi.objects_crossed.append(obj.id)
                    obj.in_time = frame_time
                    obj.active = True
                    aoi.active_time += 1
                    #print("Person " + str(obj.id) + " entered area " + str(aoi.label))
                    if aoi.label == 'customer_line':
                        obj.type = 'customer'
                
                # seeing object in are NOT for the first time
                elif aoi.polygon.contains(point):
                    if not aoi.active:
                        aoi.active_time += 1
                        
                    obj.active = True
                    aoi.active = True
                    
                    obj.waittime += 1
                





def cls2label(cls_num):
    cls_num = int(cls_num)
    if cls_num == 1:
        return 'person'
    else:
        return 'None'