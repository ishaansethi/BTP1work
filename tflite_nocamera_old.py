import os
import warnings
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import csv
import time
warnings.filterwarnings('ignore')
cwd = os.getcwd()

def preprocess_image(image_path, input_size):
    img = cv2.imread(image_path)[:,:, ::-1]
    img = img.astype(np.uint8, copy=False)
    resized_img = (np.expand_dims(cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR), axis=0)).astype(np.uint8, copy = False)
    return resized_img, img


def detect_objects(interpreter, image, threshold):
    signature_fn = interpreter.get_signature_runner()

    output = signature_fn(images=image)
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])

    results = [{'bounding_box': boxes[i], 'class_id': classes[i], 'score': scores[i]} for i in range(count) if scores[i] >= threshold]
    return results


def create_output(image_path, interpreter, threshold=0.5):
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    preprocessed_image, original_image = preprocess_image(image_path,(input_height, input_width))

    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    original_image_np = original_image.astype('uint8')
    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])

        cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), (255,0,0), 1)

        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.0f}%".format('APIS', obj['score'] * 100)
        cv2.putText(original_image_np, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    original_image_np = original_image_np.astype('uint8')
    return original_image_np, len(results)

MODEL_PATH = './model2apis.tflite'
MODEL_NAME = 'model2apis'
DETECTION_THRESHOLD = 0.01
INPUT_PATH = './frames/input{:03}.png'
OUTPUT_PATH = './frames/output{:03}.png'

bees_per_frame=[]

start = time.time()

for i in range(0,101):
    
    interpreter =tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    detection_result_image, cnt = create_output(INPUT_PATH.format(i), interpreter, threshold=DETECTION_THRESHOLD)
    
    Image.fromarray(detection_result_image).save(OUTPUT_PATH.format(i))
    
    bees_per_frame.append(cnt)
    print('Image {} done'.format(i))
    #os.remove(INPUT_PATH.format(i))
    
with open('counts', 'w') as myfile:
    wr = csv.writer(myfile, delimiter = '\n',quoting=csv.QUOTE_ALL)
    wr.writerow(bees_per_frame)
end = time.time()
print(end-start)