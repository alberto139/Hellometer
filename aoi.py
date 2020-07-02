import cv2
import json
import numpy as np
from pprint import pprint
from shapely.geometry.polygon import Polygon

def aoi(scene, mul, area_json, line_json):
    print("Importing areas of interest")
    resized_lines = []
    resized_areas = []
    aois = []
    
    # Read AoIs
    with open(area_json) as area_file:
        area_data = json.load(area_file)

   
    areas = [key['points'] for key in area_data['shapes']]
    area_names = [key['label'] for key in area_data['shapes']]

    print(len(areas))
    print(area_names)
    

    for i, line in enumerate(areas):
    
        area = areas[i]
        area_pts = [(point[0] // mul, point[1] // mul) for point in area]
        area_pts = np.array(area_pts, np.int32)
        cv2.polylines(scene,[area_pts],True,(0,255,255)) # TODO: move to utils
        resized_areas.append(area_pts)

        # Create AoI object
        aois.append(AoI(scene, area_pts, [], i, area_names[i]))


    return scene, resized_lines, resized_areas, aois

class AoI():
    def __init__(self, scene, area, line, i, label):
        self.id = i
        self.label = label
        self.relevant_classes = get_relevant_classes(self.label)
        self.line = line
        self.area = area
        self.line_color = (50,255,50)
        self.area_color = (0, 255, 255)
        self.crossed = False
        self.objects_crossed = []

        self.polygon = Polygon(self.area) # Polygon describing the area of the AoI
        self.active = False

        self.active_time = 0

def get_relevant_classes(label):
    if 'crosswalk' in label.lower():
        return [1,3]
    elif 'road' in label.lower() or 'lane' in label.lower():
        return[1,2,3,4]
    else:
        return[1,2,3,4,5]