import os
import warnings
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
#import tensorflow as tf
import csv
import time
import timeit
import sys
warnings.filterwarnings('ignore')
cwd = os.getcwd()


#gs = "videotestsrc device=/dev/video2 video/x-raw, width=640, height=480, framerate=10/1, format=(string)UYVY ! decodebin ! videoconvert ! appsink"
#cap = cv2.VideoCapture(gs, cv2.CAP_GSTREAMER)
#cap = cv2.VideoCapture("v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink")
cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)
#cap = cv2.VideoCapture(-1)

MODEL_PATH = './model2apis.tflite'
MODEL_NAME = 'model2apis'
DETECTION_THRESHOLD = 0.05
OUTPUT_PATH = './frames1/output{:03}.png'

#interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
signature_fn = interpreter.get_signature_runner()
detection_result_image, cnt = [], 0
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

#image_read = cv2.imread
image_resize = cv2.resize
expand_dims = np.expand_dims
uint8 = np.uint8
typecast = np.ndarray.astype
sqee = np.squeeze
draw_rect = cv2.rectangle
draw_text = cv2.putText
fontt = cv2.FONT_HERSHEY_SIMPLEX
#append_bees = bees_per_frame.append
save_image = cv2.imwrite

if cap.isOpened():
    #cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)
    ctr=0
    while True:
        ret_val, img = cap.read()
        start=time.time()
        if ctr%60==0:
            img = img.astype(uint8)
            resized_img = expand_dims(image_resize(img, (input_height, input_width), interpolation=cv2.INTER_LINEAR), axis=0)

            output = signature_fn(images=resized_img)
            count = int(sqee(output['output_0']))
            scores = sqee(output['output_1'])
            boxes = sqee(output['output_3'])
            results = [{'bounding_box': boxes[i], 'score': scores[i]} for i in range(count) if scores[i] >= DETECTION_THRESHOLD]

            for obj in results:
                ymin, xmin, ymax, xmax = obj['bounding_box']
                xmin = int(xmin * img.shape[1])
                xmax = int(xmax * img.shape[1])
                ymin = int(ymin * img.shape[0])
                ymax = int(ymax * img.shape[0])

                draw_rect(img, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
                y = ymin - 15 if ymin - 15 > 15 else ymin + 15
                label = "{}: {:.0f}%".format('APIS', obj['score'] * 100)
                draw_text(img,label, (xmin, y), fontt, 0.5, (0,0,0), 1)

            save_image(OUTPUT_PATH.format(ctr//60), img)
            #cv2.imshow('demo',img)
        end=time.time()
        print(end-start)
        #print(np.shape(img))
        
        #if ctr%60==0:
        #    cv2.imwrite("./frames/img{:02}.jpg".format(ctr//60),img)
        #width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        #height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        # or
        #ctr+=1
        # it gives me 0.0 :/            
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        ctr+=1
        cv2.waitKey(1)
else:
    print("camera open failed")
cap.release()
cv2.destroyAllWindows()


