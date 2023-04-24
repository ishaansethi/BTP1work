import os
import warnings
import cv2
import numpy as np
#import tflite_runtime.interpreter as tflite
import tensorflow as tf
import csv
import time
import timeit
warnings.filterwarnings('ignore')
cwd = os.getcwd()

start = time.time()
MODEL_PATH = './model2apis.tflite'
MODEL_NAME = 'model2apis'
DETECTION_THRESHOLD = 0.05
INPUT_PATH = './frames/input{:03}.png'
OUTPUT_PATH = './frames/output{:03}.png'

bees_per_frame=[]

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
signature_fn = interpreter.get_signature_runner()
detection_result_image, cnt = [], 0
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

image_read = cv2.imread
image_resize = cv2.resize
expand_dims = np.expand_dims
uint8 = np.uint8
typecast = np.ndarray.astype
sqee = np.squeeze
draw_rect = cv2.rectangle
draw_text = cv2.putText
fontt = cv2.FONT_HERSHEY_SIMPLEX
append_bees = bees_per_frame.append
save_image = cv2.imwrite

for i in range(0,101):
    img = image_read(INPUT_PATH.format(i)).astype(uint8)

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

    save_image(OUTPUT_PATH.format(i), img)

    append_bees(len(results))
    #print('Image {} done'.format(i))
    #    os.remove(INPUT_PATH.format(i))

write_csv = csv.writer
with open('counts', 'w') as myfile:
    wr = write_csv(myfile, delimiter = '\n',quoting=csv.QUOTE_ALL)
    wr.writerow(bees_per_frame)
end = time.time()
print(end-start)
