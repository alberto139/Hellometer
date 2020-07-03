import tensorflow as tf 
import cv2 
import numpy as np  
import pandas as pd
import object_class
import utils, aoi, counting
import time
import copy

import matplotlib.pyplot as plt  
import numpy as np 
from scipy import stats


# Object Detection #
# This is the main function that processes the video 
# All other methods are called from within this function but are
# defined in other files, namely utils, aoi, counting
def object_deteciton():

    #  Parameters

    mul = 2 # Factor to reduce the size of the image by
    conf_threshold = 0.5 # Ignore detections lower than this threshold
    first_detection = True # Flag for first detection, prev_detections = new_detections only when this is true
    max_id = 0 # Starting ID, all detections get assigned a ID that is sequential according to the order they are detected
    prev_detections = [] # Array to store detections from the previous frame
    customer_waittime = [] # Array to store customers sequentially in the order they leave
    area_json = 'food_counter.json' # Json with the shame of our Areas of Interest (AOIs)
    out = cv2.VideoWriter('demo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (1904, 712)) # For recording video output
    n_frames = 0 # Number of frames processed so far.
    

    # Select video source path
    cap = cv2.VideoCapture('food_counter.mov')
    #cap = cv2.VideoCapture('convenience_store.mov')

    # Read and resize the first image
    ret, img = cap.read()
    img = cv2.resize(img, (img.shape[1] // mul, img.shape[0] // mul))
    heatmap = np.zeros_like(img[:,:,0]) #Initialize heatmap to be the same shape as our first image


    # Path to trained TensorFlow model
    model_path = "frozen_inference_graph_ft2.pb" # mobilnet_fpn fine tuned on food counter data

    # Read areas of interest and creas objects of AOI class, stored in 'aois'
    img, lines, areas, aois = aoi.aoi(img, 1, area_json, [])

    # Boilerplate Tensorflow setup
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        sess = tf.Session()
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections','detection_boxes','detection_scores','detection_classes']:  # Note: might need to ad detection_masks
            tensor_name = key + ":0"
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Main loop of the process
    # As long as there is an image to be read from the video source, 
    # Process it and keep reading images
    while ret:

        # Work around for difference in frame rate from original video source
        # and screen recording of the original video
        # Assuming that the original video is 1 FPS and the screen recording is 30 FPS
        # Read but Ignore 30 consecutive frames. (Might not be exact but works well enough)
        if not n_frames % 30 == 0:
            ret, img = cap.read()
            n_frames += 1
            continue


        # Resize the image according to 'mul'
        img = cv2.resize(img, (img.shape[1] // mul, img.shape[0] // mul))

      
        # Object Detection Inference from TensorFlow model
        output_dict = sess.run(
        tensor_dict, feed_dict={
        image_tensor: np.expand_dims(img, 0)})
        output_dict['num_detections'] = int(
        output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]


        # Process output from the TensorFlow model and create objects of class 'Object' to store all relvant information
        # Some calculations such as center of the object, and bounding box location are made withing the 'Object' class.
        # All detections in the current frame are considered 'new_detections'
        new_detections = []
        for i, element in enumerate(output_dict['detection_boxes']):
            if output_dict['detection_scores'][i] > conf_threshold:
                temp_object = object_class.Object(img, output_dict['detection_boxes'][i], output_dict['detection_classes'][i])
                if not temp_object.cls_string == 'person':
                                continue

                new_detections.append(temp_object)

        # If this is the first frame consider match all new detections to themselves,
        # since they have no previous detection
        if first_detection:
            for thing in new_detections:
                thing.id = max_id
                max_id += 1
                prev_detections.append(thing)
                first_detection = False

        ### Object Traking ###
        # Matching all new detections to previous detections base on handwriten heuristic, which relies mostly on
        # pixel distance, but also considers color and shape similarity to create potential matches.
        # The best matching previous object is choosen for each new detection and considered to be the same object.
        # If a new detection does not have a potential match, it's considered to be an object we haven't seen before.
        # If a previous detection has no new match, we consider that object to no longer be in the video.
        prev_detections, max_id, customer_waittime = utils.match(prev_detections, new_detections, max_id, time.time(), customer_waittime)

        ### Analysis ###
        # Analyse which areas of interest are active acording to the center of each detected object
        counting.class_count_area(aois, prev_detections, False, time.time())


        ### Visualization ###
        draw_boxes = True
        draw_text = True
        draw_areas = True
        vis_image = np.copy(img)

        if draw_areas:
            #vis_image = utils.draw_aoi_active(vis_image, aois)
            vis_image = utils.draw_aoi_active(vis_image, aois)
        if draw_boxes:
            #img = utils.draw_boxes_active(img, prev_detections)
            vis_image = utils.draw_boxes(vis_image, prev_detections)
        if draw_text:
            #img = utils.draw_text_active(img, prev_detections)
            vis_image = utils.draw_text(vis_image, prev_detections)

        # Heat map
        heatmap = utils.update_heatmap(copy.deepcopy(heatmap), prev_detections)
        vis_heatmap = np.dstack((copy.deepcopy(heatmap), copy.deepcopy(heatmap), copy.deepcopy(heatmap)))
        try:
            vis_heatmap = vis_heatmap * 255.0 / (np.max(vis_heatmap))
            vis_heatmap = (vis_heatmap).astype(np.uint8)
        except:
            pass

        vis_heatmap = cv2.applyColorMap(vis_heatmap, cv2.COLORMAP_JET)

        
        #combined = np.hstack((vis_image, vis_heatmap))
        cv2.imshow('combined', vis_image)
        #out.write(combined)
        cv2.waitKey(1)

        # Read new image
        ret, img = cap.read()
        n_frames += 1


    cv2.imwrite('heatmap.png', vis_heatmap)

    # Graphing area utilization and customer wait times
    # This only happens at the end of the video, but it could be a seperate process
    # wich continuously updates the information. 
    station = []
    usage = []

    for area in aois:
        station.append(area.label)
        usage.append(area.active_time)

    usage = np.array(usage)
    usage = [x / (n_frames/30) for x in usage]
    fig, ax1 = plt.subplots()
    ax1.bar(station, usage)

    ax1.set_title('Percentage Occupation of Different Work Stations and Customer Line')
    ax1.set_xlabel('Station')
    ax1.set_ylabel('Time Occupied (percentage)')
    plt.show()

    fig, ax1 = plt.subplots()
    x = list(range(len(customer_waittime)))
    ax1.bar(x, customer_waittime)
    ax1.set_title('Customer Wait Times')
    ax1.set_xlabel('Customer #')
    ax1.set_ylabel('Wait Time (seconds)')
    plt.show()


# Main function call
object_deteciton()