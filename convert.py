import os
import glob
import argparse
import matplotlib
import time
import tensorflow as tf

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from tensorflow.keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt
from convertUtils import *

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/267_image.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

wkdir = os.getcwd()
# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}


# Load model into GPU / CPU
print('Loading Keras model...')
t0=time.time()
kerasModel = load_model(args.model, custom_objects=custom_objects, compile=False)
t1=time.time()
print('\nModel loaded ({0}), {1} seconds.'.format(args.model, round(t1-t0, 3)))
#print('\nkerasModel Inputs data info {0}'.format([input.op.name for input in kerasModel.inputs]))
#print('\nkerasModel Outputs data info {0}'.format([output.op.name for output in kerasModel.outputs]))

#Load Iamges
print('\nStart load test image')
inputs = load_images( glob.glob(args.input) )


print('\nStart keras inference')
t0=time.time()
kerasOutputs = predict(kerasModel, inputs)
t1=time.time()
kerasInferenceTime = round(t1-t0, 3)
print('\nKeras inference time:{0} seconds.'.format(kerasInferenceTime))


# Keras convert to tensorflow model
print('\nConverting to tensorflow2.2 model({0}).'.format("nyu.pb"))
t0=time.time()
keras_to_tensorflow(kerasModel, output_dir=wkdir, model_name="nyu.pb")
t1=time.time()
print('\nFinish converting to tensorflow2.2 model({0}), {1} seconds.'.format("nyu.pb", round(t1-t0, 3)))


###!!!!!! reload tensorflow 2.2 model to modify input shape and save !!!!##############
tensorflow_reshape_input_shape(wkdir, "nyu.pb", reshape=(1, 480, 640, 3), reshape_tensor_name='x', reshape_dtype='float32')

#Writ tensorflow lite graph summary
summary_tensorflow_grpah(wkdir, "nyu.pb")

# After resave nyu.pb, reload it for inference
tensorflowOutputs, tensorflowInferenceTime = tensorflow_inference(wkdir, "nyu.pb", 'import/x:0', 'import/import/Identity:0', inputs)

# convert to tensorflow lite model and inference
convert_to_tensorflow_lite(wkdir, "nyu.pb", "nyu.tflite", 'import/x:0', 'import/import/Identity:0')
tensorflowLiteOutputs, tensorflowLiteInferenceTime = tensorflow_lite_inference(wkdir, "nyu.tflite", inputs, 'float32')


#Start converting to tensorflow-lite quantize model and inference
convert_to_tensorflow_lite(wkdir, "nyu.pb", "nyuQuan.tflite", 'import/x:0', 'import/import/Identity:0', [tf.lite.Optimize.DEFAULT])
tensorflowLiteQuanOutputs, tensorflowLiteQuanInferenceTime = tensorflow_lite_inference(wkdir, "nyuQuan.tflite", inputs, 'float32')

#Reverse Tensorflow and Lite depth map
tensorflowOutputs = reverse_depth_value(tensorflowOutputs)
tensorflowLiteOutputs = reverse_depth_value(tensorflowLiteOutputs)
tensorflowLiteQuanOutputs = reverse_depth_value(tensorflowLiteQuanOutputs)

#Plot Keras, Tensorflow, Tensorflow-lite depth map
vizKeras = display_images(outputs = kerasOutputs, inputs = inputs)
vizTensorflow = display_images(outputs = tensorflowOutputs, inputs = inputs)
vizTensorflowLite = display_images(outputs = tensorflowLiteOutputs, inputs = inputs)
vizTensorflowLiteQuan = display_images(outputs = tensorflowLiteQuanOutputs, inputs = inputs)
f, axarr = plt.subplots(4,1)
for ax in axarr.flat:
    ax.axis("off")
axarr[0].text(0.5, 0.5, "keras inference "+str(kerasInferenceTime)+" seconds", fontsize=10, ha='left')
axarr[1].text(0.5, 0.5, "tensorflow inference "+str(tensorflowInferenceTime)+" seconds", fontsize=10, ha='left')
axarr[2].text(0.5, 0.5, "tensorflow-lite inference "+str(tensorflowLiteInferenceTime)+" seconds", fontsize=10, ha='left')
axarr[3].text(0.5, 0.5, "tensorflow-lite Quantize inference "+str(tensorflowLiteQuanInferenceTime)+" seconds", fontsize=10, ha='left')

axarr[0].imshow(vizKeras)
axarr[1].imshow(vizTensorflow)
axarr[2].imshow(vizTensorflowLite)
axarr[3].imshow(vizTensorflowLiteQuan)

plt.savefig('result.png')
plt.show()