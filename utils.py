# tracking.py

import numpy as np  
import math
#import video_functions
import cv2
from matplotlib import pyplot as plt
import time
#import fire_push


def match(prev_detections, new_detections, max_id, frame_time, customer_waittime):
    distance_threshold = 200 # what to make 250
    missing_threshold = 0 #8 # 10
    difference_threshold= 750 #2000 #82
    histogram_threshold = 0.20
    match_score_threshold = 70000
    test = False
    size = 64
    very_close_threshold = 20
    min_size_threshold = 100000

    # PEDESTRAIN
    ped_distance_threshold = 50 # what to make 250
    ped_difference_threshold= 45 #2000 #82
    ped_histogram_threshold = 0.50


    # Existing Match
    for prev in prev_detections:
        #print("------------------------ PREV ID: " + str(prev.id) +str("--------------------------------------------"))
        #print("1. " + str(prev.prev_centers))
        potential_matches = [] # list of potenital matches between the current prev and all the new detections
        potential_matches_distance = []
        potential_matches_difference = []
        match = None

        if not prev.match:
            for new in new_detections:
                if not new.match and not prev.match:
                    # TODO: Euclidian distance between prev_object.center and new_object.center
                    distance = math.sqrt( (prev.x - new.x)**2 + (prev.y - new.y)**2 )
                    # CHECK SIMILARITY
                    new_resized = cv2.resize(new.subimg, (size, size)) 
                    prev_resized = cv2.resize(prev.subimg, (size, size)) 
                    result, difference_score = difference(new_resized, prev_resized)
                    
                    #result, difference_score = similarity(new, prev)
                    
                    # Similarity Algorithm using histograms
                    new_gray = cv2.cvtColor(new_resized, cv2.COLOR_BGR2GRAY)
                    prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_BGR2GRAY)
                    new_hist = cv2.calcHist([new_gray], [0], None, [256], [0,256])
                    prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0,256])

                    hist_comp = cv2.compareHist(new_hist, prev_hist, cv2.HISTCMP_CORREL)
                    
  

                
                    if (distance <= distance_threshold ) and (difference_score < difference_threshold) and (prev.cls == new.cls) and (abs(hist_comp) > histogram_threshold):
                    
                        # If true, add to potential match. Then evaluate for best match
                        potential_matches.append(new)
                        potential_matches_distance.append(distance)
                        potential_matches_difference.append(difference_score)
                    #prev.match = True

                    if test and not potential_matches: # and prev.active:
                        print("-------NO MATCH-------")
                        #cv2.imshow("no prev", prev.subimg)
                        #cv2.imshow("no new", new.subimg)
                        #cv2.imshow('difference', result)

                        print("ID: " + str(prev.id))
                        print("difference: " + str(difference_score))
                        print("distance: " + str(distance))
                        print("hist: " + str(hist_comp))
                        #print("score: " + str(match_score))
                        #print("class: " + str(prev.cls))
                        #print('conf:' + str(match.conf))

                        #cv2.waitKey(0)

            
            best_match_score = float("inf") # Lower is better
            #print('len potential matches: ' + str(len(potential_matches)))
            #match = None
            for i, potential_match in enumerate(potential_matches):
                match_score = potential_matches_distance[i] * potential_matches_difference[i]
                if match_score < best_match_score and match_score < match_score_threshold:
                    
                    best_match_score = match_score
                    match = potential_match
                    match_difference = potential_matches_difference[i]
                    match_distance = potential_matches_distance[i]


            
                #pass

        if match:     

            if test: # and prev.active:
                print("-------MATCH-------")
                #cv2.imshow("prev", prev.subimg)
                #cv2.imshow("new", match.subimg)
                #cv2.imshow('difference', result)

                print("ID: " + str(prev.id))
                print("difference: " + str(match_difference))
                print("distance: " + str(match_distance))
                print("hist: " + str(hist_comp))
                print("score: " + str(match_score))
                print("class: " + str(prev.cls))
                print('conf:' + str(match.conf))

                #cv2.waitKey(0)
                #pass


            # TODO: Replace this with an object update function
            prev.box = match.box
            (prev.ymin, prev.xmin, prev.ymax, prev.xmax) = match.box
            prev.frame = match.frame
            prev.subimg = match.frame[match.ymin:match.ymax, match.xmin:match.xmax]
            prev.x = int(match.xmin + (match.xmax - match.xmin) / 1.5)
            prev.y = int(match.ymin + (match.ymax - match.ymin) / 1.5)
            prev.center = (match.x, match.y)
            prev.prev_centers.append(match.center)
            prev.prev_times.append(frame_time)
            prev.match = True
            match.match =  True
            prev.conf = match.conf
            #prev.color = match.color

            prev.last_detected = time.time()

            prev.missing = 0
            pass

                        
    # No Match for prev. Increade missing counter and remove if missing for a certain amount
    for prev in prev_detections:
        if not prev.match:
            prev.missing += 1
            prev.active = False
        if prev.missing > missing_threshold or (prev.crossed and prev.missing > 3) :
            if prev.type == 'customer' and prev.waittime >= 10:
                print(str(prev.id) + " wait time: " + str(prev.waittime))
                customer_waittime.append(prev.waittime)

            prev_detections.remove(prev)

    # No Match for new
    for new in new_detections:
        if not new.match:
            new.id = max_id
            max_id +=1
            #print("increased max id")
            #print("Added: " + str(new.id))
            prev_detections.append(new)
        

    # Reset match for the next frame                    
    for prev in prev_detections:
        prev.match = False

    

    return prev_detections, max_id, customer_waittime



def similarity(prev, new):

    test = False
    #blur = cv2.blur(img,(5,5))
    prev_image = cv2.blur(prev.subimg, (10, 10))
    new_image = cv2.blur(new.subimg, (10, 10))

    prev_x, prev_y = prev_image.shape[1], prev_image.shape[0]
    new_x, new_y = new_image.shape[1], new_image.shape[0]


    prev_area = prev_y * prev_x
    new_area = prev_y * prev_x

    max_x =  max(new_image.shape[1], prev_image.shape[1])
    max_y =  max(new_image.shape[0], prev_image.shape[0])


    prev_pad_x = max(max_x - prev_x, 0) // 2
    prev_pad_y = max(max_y - prev_y, 0) // 2
    new_pad_x = max(max_x - new_x, 0) // 2
    new_pad_y = max(max_y - new_y, 0) // 2

    #cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
    #image = cv2.copyMakeBorder( src, top, bottom, left, right, borderType)

    prev_image = cv2.copyMakeBorder( prev_image, prev_pad_y, prev_pad_y, prev_pad_x, prev_pad_x, cv2.BORDER_CONSTANT, value=(255,255,255))
    new_image = cv2.copyMakeBorder( new_image, new_pad_y, new_pad_y, new_pad_x, new_pad_x, cv2.BORDER_CONSTANT, value=(255,255,255))
    #cv2.imshow("padding test prev", prev_image)
    #cv2.imshow("padding test new", new_image)
    #cv2.waitKey(0)



    # The new image is larger. Resize to the new image
    if prev_area < new_area:
        prev_image = cv2.resize(prev_image, (new_image.shape[1], new_image.shape[0])) 
        area = new_area
    else:
        new_image = cv2.resize(new_image, (prev_image.shape[1], prev_image.shape[0]))
        area = prev_area

    result, difference_score = difference(new_image, prev_image)
    #print("new_differance_score:" + str(difference_score))
    resut = result / (max_x * max_y)
    return result, difference_score


def speed_estimation(detections):

    for obj in detections:
        if len(obj.prev_centers) >= 2:
            p1 = obj.prev_centers[-1]
            p2 = obj.prev_centers[-2]

            distance_px = math.sqrt((p2[0] - p1[0])**2  +  (p2[1] - p1[1])**2 )
            time_distance = abs(obj.prev_times[-1] - obj.prev_times[-2])
            mph = (distance_px / time_distance) / (17.6 * 2)  #assuming 1 px = 2 inch
            #print('distance: ' + str(distance))
            #print('time_distance: ' + str(time_distance))
            #print('speed: ' + str(speed))
            obj.speed = mph

    return detections

def non_max_suppression_fast(boxes):
    overlapThresh = 0.5
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    idxs = sorted(idxs, reverse = True)
    
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                                np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    pick = sorted(pick)
    idxs = sorted(idxs)

    return pick

    return boxes[pick].astype("int")

def print_fps(start_time, total, count, disp = False, name = 'thread'):
    fps = (1.0 / (time.time() - start_time))
    total += fps
    count += 1
    avg = total / count
    if disp:
        #print("Capture FPS: " + str(fps))
        #print("Total FPS: " + str(total))
        #print("Counts FPS: " + str(count))
        print(str(name) + " FPS: " + str(avg))
    start_time = time.time()
    return fps, start_time, total, count, avg

def fps(start_time, count, avg_fps, name, disp = False):
    count += 1
    fps = 1.0 / (time.time() - start_time)
    avg_fps += fps

   

    if count % 10 == 0 and disp:
        print('------------------')
        print(name)
        print("count: " + str(count))
        print("fps: " + str(fps))
        print("avg_fps: " + str(avg_fps / count))
        avg_fps = int(str(avg_fps / count))
    start_time = time.time()
    return start_time, count, avg_fps

def draw_paths(frame, detections):

    color = (50,50,255)

    for obj in detections:
        
        centers = obj.prev_centers
        centers.reverse()

        #centers = [np.array(centers[0:15], np.int32)]
        #cv2.polylines(frame,centers,True,(50,255,50), thickness = 2)
        if len(centers) > 2 :#and obj.id == 2894:
            for index, point in enumerate(centers):
                if index == len(centers) -1:
                    #print("not enough points")
                    break
                cv2.line(frame, point, centers[index+1], color, 2)
                #print("THE LINE HAS BEEN DRAWN")

            
    return frame 


def draw_paths_active(frame, detections):

    color = (50,50,255)

    for obj in detections:
        if not obj.active:
            continue
        centers = obj.prev_centers
        centers.reverse()

        #centers = [np.array(centers[0:15], np.int32)]
        #cv2.polylines(frame,centers,True,(50,255,50), thickness = 2)
        if len(centers) > 2 :#and obj.id == 2894:
            for index, point in enumerate(centers):
                if index == len(centers) -1:
                    #print("not enough points")
                    break
                cv2.line(frame, point, centers[index+1], color, 2)
                #print("THE LINE HAS BEEN DRAWN")

            
    return frame 

def difference(img1, img2):
    """
    Calculates the difference between two images.
    img1: numpy array
    img2: numpy array
    Returns: new image displaying the differences (high pixel values = very different)
    """
    viz = False

    # Grayscale the two images
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype('float32')
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype('float32')
    
    # Normalize
    gray1 /= 255
    gray2 /= 255


    # Get the difference in magnitude for every pixel. int16 so that negative don't wrap around
    difference = abs(np.subtract(gray1, gray2))

    result = difference
    # Make all the pixels under a threshold zero
    # NOTE: This works but makes everthing really slow

    #result = np.array([[0 if pixel < 0.2 else pixel for pixel in column] for column in difference])
    result[result < 0.1] = 0 # Same thing as the line above but WAY faster. I <3 numpy
    

    if viz:  
        cv2.imshow('difference', result)#.astype(np.int8))
        cv2.waitKey(0)

    score = sum(sum(result)) # Sum all the pixels together. Similar frames will have a low score

    return result, score


#vis_util.visualize_boxes_and_labels_on_image_array(frame, output_dict['detection_boxes'],output_dict['detection_classes'],output_dict['detection_scores'],category_index,use_normalized_coordinates=True, min_score_thresh = .2, line_thickness=4)
def draw_text(image, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.30
    margin = 5
    thickness = 1
    color = (255, 255, 255)
    line = 1
    text = "sample"
    overlay = image.copy()
    output = image.copy()
    alpha = .70
    
    size = cv2.getTextSize(text, font, font_scale, thickness)
    
    text_width = size[0][0]
    text_height = size[0][1]
    line_height = text_height + size[1] + margin
    
    for detection in detections:
        x = detection.xmin
        y = detection.ymin - 35
        cv2.rectangle(image, (x, y), (x + 60, y + 40),(0, 0, 0), -1)
        y = detection.ymin - 30
        #cv2.addWeighted(overlay, alpha, image, 1 - alpha,0, image)
        cv2.putText(image, cls2label(detection.cls), (x + margin, y), font, font_scale, color, thickness)
        cv2.putText(image, "id: " + str(detection.id), (x + margin, y + 15), font, font_scale, color, thickness)
        cv2.putText(image, "conf: " + str(detection.conf)[:4], (x + margin, y + 25), font, font_scale, color, thickness)
        #cv2.putText(image, "speed: " + str(int(detection.speed)), (x + margin, y + 40), font, font_scale, color, thickness)
        #cv2.putText(image, "direction: " + str(detection.direction), (x + margin, y + 60), font, font_scale, color, thickness)
        
    return image


def draw_text_active(image, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.30
    margin = 5
    thickness = 1
    color = (255, 255, 255)
    line = 1
    text = "sample"
    overlay = image.copy()
    output = image.copy()
    alpha = .70
    
    size = cv2.getTextSize(text, font, font_scale, thickness)
    
    text_width = size[0][0]
    text_height = size[0][1]
    line_height = text_height + size[1] + margin
    
    for detection in detections:
        if not detection.active:
            continue
        x = detection.xmax
        y = detection.ymin
        cv2.rectangle(image, (x, y), (x + 40, y + 20),(0, 0, 0), -1)
        y = detection.ymin + 15
        #cv2.addWeighted(overlay, alpha, image, 1 - alpha,0, image)
        cv2.putText(image, cls2label(detection.cls), (x + margin, y), font, font_scale, color, thickness)
        cv2.putText(image, "id: " + str(detection.id), (x + margin, y + 5), font, font_scale, color, thickness)
        #cv2.putText(image, "speed: " + str(int(detection.speed)), (x + margin, y + 40), font, font_scale, color, thickness)
        #cv2.putText(image, "direction: " + str(detection.direction), (x + margin, y + 60), font, font_scale, color, thickness)
        
    return image


def draw_boxes(frame, detections):
    test = False
    line_x = 0
    thickness = 2
    for thing in detections:
        
        if thing.missing > 0:
            pass
        # Draw box
        cv2.rectangle(frame, (thing.xmin, thing.ymin), (thing.xmax, thing.ymax), (255,0,0), thickness)
        #frame = cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 2)
        # Draw center
        # frame = draw_points(frame, thing)
        # draw line connecting the previous centers
        frame = draw_center_line(frame, line_x)
        # draw ID
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame,str(thing.id),thing.center, font, 0.5,(255,255,255),1,cv2.LINE_AA)
        #print(thing.id)

        if test:
            cv2.imshow(str(thing.id), thing.subimg)
            cv2.waitKey(1)
    
    #cv2.imshow('detections', frame)
    #cv2.waitKey(1)

    return frame

def draw_boxes(frame, detections):
    test = False
    line_x = 0
    thickness = 2
    for thing in detections:
        
        if thing.missing > 0:
            pass
        # Draw box
        # vehicle, pedestrian, sign?, cyclist
        color = [(255,255,0), (0,255,0), (0,0,255), (255,0,255)]
        try:
            #cv2.rectangle(frame, (thing.xmin, thing.ymin), (thing.xmax, thing.ymax), color[thing.cls - 1], 1)
            cv2.rectangle(frame, (thing.xmin, thing.ymin), (thing.xmax, thing.ymax), thing.color, 1)
        except:
            cv2.rectangle(frame, (thing.xmin, thing.ymin), (thing.xmax, thing.ymax), (255,255,255), 3)
            cv2.circle(frame, thing.center, 2, thing.color, -1)      
        # draw line connecting the previous centers
        frame = draw_center_line(frame, line_x)
        # draw ID
        font = cv2.FONT_HERSHEY_SIMPLEX

        if test:
            cv2.imshow(str(thing.id), thing.subimg)
            cv2.waitKey(1)


    return frame

def draw_boxes_active(frame, detections):
    test = False
    line_x = 0
    thickness = 2
    for thing in detections:
        if not thing.active:
            continue
        if thing.missing > 0:
            pass
        # Draw box
        color = [(255,255,0), (0,255,0), (0,0,255), (255,0,255)]
        try:
            #cv2.rectangle(frame, (thing.xmin, thing.ymin), (thing.xmax, thing.ymax), color[thing.cls - 1], 1)
            cv2.rectangle(frame, (thing.xmin, thing.ymin), (thing.xmax, thing.ymax), thing.color, 1)
            cv2.circle(frame, thing.center, 2, thing.color, -1)         
        
        except:
            cv2.rectangle(frame, (thing.xmin, thing.ymin), (thing.xmax, thing.ymax), (255,255,255), 3)
        frame = draw_center_line(frame, line_x)
        # draw ID
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame,str(thing.id),thing.center, font, 0.5,(255,255,255),1,cv2.LINE_AA)
        #print(thing.id)

        if test:
            cv2.imshow(str(thing.id), thing.subimg)
            cv2.waitKey(1)
    
    #cv2.imshow('detections', frame)
    #cv2.waitKey(1)

    return frame


# Draw area poligons
def draw_lines(img, aois):
    
    
    for aoi in aois:
        line = aoi.line
        point_1 = (line[0][0], line[0][1])
        point_2 = (line[1][0], line[1][1])
        if aoi.crossed:
            color = (0, 100, 255)
            aoi.crossed = False
            cv2.line(img, point_1, point_2, color, 3)
        else:
            color = (255, 255, 0)
            cv2.line(img, point_1, point_2, color, 3)
        
        
    return img

def draw_lines_active(img, aois):
    
    
    for aoi in aois:
        if not aoi.active:
            continue
        line = aoi.line
        point_1 = (line[0][0], line[0][1])
        point_2 = (line[1][0], line[1][1])
        if aoi.crossed:
            color = (0, 100, 255)
            aoi.crossed = False
            cv2.line(img, point_1, point_2, color, 3)
        else:
            color = (255, 255, 0)
            cv2.line(img, point_1, point_2, color, 3)
        
        
    return img

def draw_aoi(img, aois):
    for aoi in aois:
        area = aoi.area
        pts = [(point[0], point[1]) for point in area]
        pts = np.array(pts, np.int32)
        cv2.polylines(img,[pts],True,(0,255,255), thickness = 2) 
    return img

def draw_aoi_active(img, aois):
    for aoi in aois:
        if not aoi.active:
            continue
        area = aoi.area
        pts = [(point[0], point[1]) for point in area]
        pts = np.array(pts, np.int32)
        #cv2.polylines(img,[pts],True,(50,255,50), thickness = 2) 
        cv2.polylines(img,[pts],True,aoi.line_color, thickness = 2) 
    return img

def write_counts(frame, counts):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame,(10,35),(500,80),(0,0,0), cv2.FILLED)
    cv2.putText(frame,str(counts),(20, 50), font, 0.5,(0,255,0),1,cv2.LINE_AA, thinkness = 2)
    
    #print("here we go again")
    #cv2.imshow('detections', frame)
    #cv2.waitKey(1)
    return frame

def draw_center_line(frame, line_x):
    color = (0,0,0)
    frame = cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), color)
    return frame

def rotate(frame):
    if rotate:

        frame = np.rot90(frame)
        frame = frame.copy() # This has to be here beacuse rot90 causes a bug otherwise

        return frame

def epoch_to_human(epoch):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))


def cls2label(cls_num):
    cls_num = int(cls_num)
    if cls_num == 1:
        return 'person'
    elif cls_num == 2:
        return 'car'
    elif cls_num == 3:
        return 'cyclist'
    elif cls_num == 4:
        return 'bus'
    else:
        return 'idk_what_this_is'

def video_dimensions(size):
    if size == '144p':
        return (256, 144)
    if size == '240p':
        return (426, 240)
    if size == '360p':
        return (640, 360)
    if size == '480p':
        return (854, 480)
    if size == '720p':
        return (1280, 720)
    if size == '1080p':
        return (1920, 1080)
    if size == '4k':
        return (4096, 2160)
    if size == 'caltrans':
        return (352, 240)
    if size == 'nevada':
        return (360, 240)
    if size == 'sacramento':
        return (720, 480)


# TODO: for larger loops fill a temp dataframe using iloc, then append to permanent dataframe and clear temp every 1000 iterations
def record_df(df, obj):
    #col_names =  ['timestamp', 'id', 'class', 'aoi_id', 'speed']
    df.loc[len(df)] = [obj.str_time, obj.id, cls2label(obj.cls), obj.aoi_ids, obj.speed, obj.directions, obj.prev_centers]

    #serve_charts(df)
    df.to_csv('test.csv', index = False)

def epoch_to_human(epoch):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))

# Convert from human readable time stamp to epoch time
# human readable time stamp should be in the following format:
# date_time = '29.08.2011 11:05:02'
def human_to_epoch(date_time):
    pattern = '%d.%m.%Y %H:%M:%S'
    epoch = int(time.mktime(time.strptime(date_time, pattern)))
    return epoch


def update_heatmap(heatmap, detections):

    for detection in detections:
        # subimg = self.inputs[r:r+self.kernel_size, c:c+self.kernel_size, :]
        #if detection.active:
        heatmap[detection.y-30:detection.y+30, detection.x-20:detection.x+20]+= 1

    heatmap = heatmap * .999
    #print("max: " + str(np.max(heatmap)))
    #print("min: " + str(np.min(heatmap)))
    return heatmap
