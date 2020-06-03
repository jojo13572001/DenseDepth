import os
import time
import numpy as np
import tensorflow as tf

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from tensorflow.python.platform import gfile
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def keras_to_tensorflow(keras_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
 
    if os.path.exists(output_dir) == False:
       os.mkdir(output_dir)

    full_model = tf.function(lambda x: keras_model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    #layers = [op.name for op in frozen_func.graph.get_operations()]
    tf.io.write_graph(frozen_func.graph, output_dir, name=model_name, as_text=False)


def tensorflow_reshape_input_shape(path, model_name, reshape, reshape_tensor_name, reshape_dtype):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        # load model from pb file
        filePath = os.path.join(path, model_name)
        with gfile.FastGFile(filePath,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            f.close()
            sess.graph.as_default()
            #modify input shape to be fixed size, not NONE
            inputReshape = tf.compat.v1.placeholder(shape=reshape, name=reshape_tensor_name, dtype=reshape_dtype )
            inputIndex = reshape_tensor_name+':0'
            tf.import_graph_def(graph_def, input_map={inputIndex: inputReshape})
            tf.io.write_graph(sess.graph, path, name=model_name, as_text=False)
        sess.close()
        # print all operation names 
        #print('\n===== ouptut operation names =====\n')
        #for op in sess.graph.get_operations():
          #print('\n'+op.name)
        #print('\nInputs data info {0}'.format([input.op.name for input in sess.graph.inputs]))
        #print('\nOutputs data info {0}'.format([output.op.name for output in sess.graph.outputs])) 

def tensorflow_inference(path, model_name, input_tensor_name, output_tensor_name, inputs):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        # load model from pb file
        filePath = os.path.join(path, model_name)
        with gfile.FastGFile(filePath,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            f.close()
            sess.graph.as_default()
            tf.import_graph_def(graph_def)

        # inference by the model (op name must comes with :0 to specify the index of its output)
        tensor_input = sess.graph.get_tensor_by_name(input_tensor_name)
        tensor_output = sess.graph.get_tensor_by_name(output_tensor_name)
        print('\nStart tensorflow2.2 inference')
        t0=time.time()
        outputs = sess.run(tensor_output, {tensor_input: inputs})
        t1=time.time()
        inferenceTime = round(t1-t0, 3)
        print('\nTensorflow2.2 inference time:{0} seconds.'.format(inferenceTime))
        sess.close()
        return outputs, inferenceTime

def convert_to_tensorflow_lite(path, tensorflowLite_model_name, model_name, input_tensor_name, output_tensor_name, quantize=[]):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        # load model from pb file
        filePath = os.path.join(path, tensorflowLite_model_name)
        with gfile.FastGFile(filePath,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def)

        # inference by the model (op name must comes with :0 to specify the index of its output)
        tensor_input = sess.graph.get_tensor_by_name(input_tensor_name)
        tensor_output = sess.graph.get_tensor_by_name(output_tensor_name)
        #Start converting to tensorflow-lite model
        print('\nStart converting to tensorflow lite')
        t0=time.time()
        converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, [tensor_input], [tensor_output])
        converter.optimizations = quantize
        tfliteModel = converter.convert()
        t1=time.time()
        print('\nTensorflow lite converting time:{0} seconds.'.format(round(t1-t0, 3)))
        t0=time.time()
        open(model_name, "wb").write(tfliteModel)
        t1=time.time()
        print('\nTensorflow lite saving model time:{0} seconds.'.format(round(t1-t0, 3)))
        sess.close()

def tensorflow_lite_inference(path, model_name, inputs, inputs_astype):
    # Load TFLite model and allocate tensors.
    filePath = os.path.join(path, model_name)
    interpreter = tf.compat.v1.lite.Interpreter(filePath)
    interpreter.allocate_tensors()

    # Start tensorflow lite inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    inputs = inputs.astype(inputs_astype)
    interpreter.set_tensor(input_details[0]['index'], inputs)
    print('\nStart tensorflowLite inference')
    t0=time.time()
    interpreter.invoke()
    t1=time.time()
    inferenceTime = round(t1-t0, 3)
    print('\nTensorflowLite inference time:{0} seconds.'.format(inferenceTime))
    outputs = interpreter.get_tensor(output_details[0]['index'])
    return outputs, inferenceTime

def reverse_depth_value(depthMap):
    maxValue = np.max(depthMap)
    minValue = np.min(depthMap)
    return  (maxValue -  depthMap + minValue)

def summary_tensorflow_grpah(path, model_name):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        # load model from pb file
        filePath = os.path.join(path, model_name)
        with gfile.FastGFile(filePath,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            f.close()
            sess.graph.as_default()
            tf.import_graph_def(graph_def)
            tf.compat.v1.summary.FileWriter('log',graph=sess.graph)
        sess.close()