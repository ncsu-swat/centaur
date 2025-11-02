
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_image_transpose_inputs():
    list_of_inputs = []
    
    # Input 1: 3D tensor (height, width, channels)
    image_3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    input_dict = {
        "image": tf.constant(image_3d),
        "name": "transpose_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 4D tensor (batch, height, width, channels)
    image_4d = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_4d),
        "name": "transpose_4d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 3D tensor with negative values
    image_3d_neg = np.array([[[1, -2, 3], [-4, 5, -6]], [[7, -8, 9], [10, -11, 12]]], dtype=np.float32)
    input_dict = {
        "image": tf.constant(image_3d_neg),
        "name": "transpose_3d_neg"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with different channel count
    image_4d_diff_channels = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]], dtype=np.float32)
    input_dict = {
        "image": tf.constant(image_4d_diff_channels),
        "name": "transpose_4d_diff_channels"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 3D tensor with floating point values
    image_3d_float = np.array([[[1.5, 2.7], [3.9, 4.1]], [[5.2, 6.8], [7.3, 8.6]]], dtype=np.float32)
    input_dict = {
        "image": tf.constant(image_3d_float),
        "name": "transpose_3d_float"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 4D tensor with single channel
    image_4d_single_channel = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_4d_single_channel),
        "name": "transpose_4d_single_channel"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D tensor with single channel
    image_3d_single_channel = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_3d_single_channel),
        "name": "transpose_3d_single_channel"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 4D tensor with different shapes
    image_4d_diff_shape = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_4d_diff_shape),
        "name": "transpose_4d_diff_shape"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with zero values
    image_3d_zero = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_3d_zero),
        "name": "transpose_3d_zero"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with large values
    image_4d_large_values = np.array([[[[100, 200], [300, 400]], [[500, 600], [700, 800]]]], dtype=np.int32)
    input_dict = {
        "image": tf.constant(image_4d_large_values),
        "name": "transpose_4d_large_values"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.image.transpose"] = tf_image_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.transpose'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.transpose', generated_inputs['tf.image.transpose'], lib="tf", suffix=0)
