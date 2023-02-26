#import numpy as np
import tensorflow as tf
import numpy as np
import cv2 as cv
import os 


def img_process(img):
  img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  # Setting parameter values
  t_lower = 50  # Lower Threshold
  t_upper = 150  # Upper threshold
  aperture_size = 3
  
  # Applying the Canny Edge filter
  edge = (cv.Canny(np.uint8(img_gray), t_lower, t_upper))
  img = (cv.cvtColor(edge,cv.COLOR_GRAY2RGB)).astype(float)
  return img

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="D:/Satellite/tflite_quant_model_2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

dir_path = '/test/bad/0.jpg'
count=0
for i in os.listdir(dir_path):
    img = cv.imread(dir_path+'//' + i)
    img = img_process(img)

    X = img_to_array(img)
    X = np.expand_dims(X, axis = 0)
    images = np.vstack([X])
    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = images
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

#Gap





