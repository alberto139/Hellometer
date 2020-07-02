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

def object_deteciton():

    # hyper parameters
    mul = 2 # Factor to reduce the size of the image by
    conf_threshold = 0.5 # make lower for convenience store
    first_detection = True
    max_id = 0
    prev_detections = []
    area_json = 'food_counter.json'
    # Pandas dataframe setup
    col_names =  ['timestamp', 'id', 'class', 'aoi_id', 'speed', 'direction']
    df = pd.DataFrame(columns = col_names)
    customer_waittime = []
    out = cv2.VideoWriter('demo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (1904, 712))
    

    # Get video source
    cap = cv2.VideoCapture('food_counter.mov')
    #cap = cv2.VideoCapture('convenience_store.mov')
    ret, img = cap.read()
    img = cv2.resize(img, (img.shape[1] // mul, img.shape[0] // mul))
    heatmap = np.zeros_like(img[:,:,0])


    # detection model path
    model_path = "frozen_inference_graph_ft2.pb" # mobilnet_fpn COCO only

    # AOIs
    img, lines, areas, aois = aoi.aoi(img, 1, area_json, [])

    # Tensorflow setup
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

    # Process each video frame
    prev_img = np.array(0)
    n_frames = 1
    while ret:

        if not n_frames % 30 == 0:
            ret, img = cap.read()
            n_frames += 1
            continue


        # Resize the image to something more manageble
        img = cv2.resize(img, (img.shape[1] // mul, img.shape[0] // mul))

        #cv2.imwrite('/home/alberto/Desktop/hellometer/hellometer_images/' + str(time.time()) + '.jpeg', img)

        

        # Run inference with the tensorflow model on the current image
        output_dict = sess.run(
        tensor_dict, feed_dict={
        image_tensor: np.expand_dims(img, 0)})
        output_dict['num_detections'] = int(
        output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]


        # Process detections into Object class
        new_detections = []
        for i, element in enumerate(output_dict['detection_boxes']):
            if output_dict['detection_scores'][i] > conf_threshold:
                temp_object = object_class.Object(img, output_dict['detection_boxes'][i], output_dict['detection_classes'][i])
                if not temp_object.cls_string == 'person':
                                continue

                new_detections.append(temp_object)

        #prev_detections = new_detections # TODO: Replace with matching
        if first_detection:
            for thing in new_detections:
                thing.id = max_id
                max_id += 1
                prev_detections.append(thing)
                first_detection = False

        prev_detections, max_id, customer_waittime = utils.match(prev_detections, new_detections, max_id, time.time(), customer_waittime)

        # activate if in area
        counting.class_count_area(aois, prev_detections, df, False, time.time())


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
        #heatmap = heatmap * .90
        vis_heatmap = np.dstack((copy.deepcopy(heatmap), copy.deepcopy(heatmap), copy.deepcopy(heatmap)))
        try:
            vis_heatmap = vis_heatmap * 255.0 / (np.max(vis_heatmap))
            vis_heatmap = (vis_heatmap).astype(np.uint8)
        except:
            pass
        #print(np.max(vis_heatmap))
        #print(vis_heatmap.shape)
        vis_heatmap = cv2.applyColorMap(vis_heatmap, cv2.COLORMAP_JET)
        #print(np.max(heatmap))



        for area in aois:
            #print('-----' + str(area.label) + '-----')
            #print('occupied: ' + str(area.active_time))
            pass


        # Show the processed image
        #cv2.imshow("food_counter", vis_image)
        #cv2.imshow("heatmap", vis_heatmap)
        
        combined = np.hstack((vis_image, vis_heatmap))
        cv2.imshow('combined', combined)
        #print(combined.shape)
        out.write(combined)
        cv2.waitKey(1)

        # Read new image
        prev_img = img
        ret, img = cap.read()


        n_frames += 1

    ### Analysis  Visualizations ###
    cv2.imwrite('heatmap.png', vis_heatmap)

    # Area usage
    station = []
    usage = []

    for area in aois:
        station.append(area.label)
        usage.append(area.active_time)

    usage = np.array(usage)
    usage = [x / (n_frames/30) for x in usage]
    fig, ax1 = plt.subplots()
    ax1.bar(station, usage)
    #ax1.plot(x, y_line, color='red')
    ax1.set_title('Percentage Occupation of Different Work Stations and Customer Line')
    ax1.set_xlabel('Station')
    ax1.set_ylabel('Time Occupied (percentage)')
    plt.show()

    fig, ax1 = plt.subplots()
    x = list(range(len(customer_waittime)))
    ax1.bar(x, customer_waittime)
    #ax1.plot(x, y_line, color='red')
    ax1.set_title('Customer Wait Times')
    ax1.set_xlabel('Customer #')
    ax1.set_ylabel('Wait Time (seconds)')
    plt.show()



    # Customer wait times


object_deteciton()