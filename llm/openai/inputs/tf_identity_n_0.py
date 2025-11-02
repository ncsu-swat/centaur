
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_identity_n_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor
    input_tensor_1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict_1 = {
        "input": [tf.constant(input_tensor_1)],
        "name": "test_1"
    }
    list_of_inputs.append(input_dict_1)
    
    # Input 2: 3D tensor
    input_tensor_2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict_2 = {
        "input": [tf.constant(input_tensor_2)],
        "name": "test_2"
    }
    list_of_inputs.append(input_dict_2)
    
    # Input 3: 1D tensor with negative values
    input_tensor_3 = np.array([-1, -2, -3], dtype=np.int64)
    input_dict_3 = {
        "input": [tf.constant(input_tensor_3)],
        "name": "test_3"
    }
    list_of_inputs.append(input_dict_3)
    
    # Input 4: Scalar tensor
    input_tensor_4 = np.array(42, dtype=np.float64)
    input_dict_4 = {
        "input": [tf.constant(input_tensor_4)],
        "name": "test_4"
    }
    list_of_inputs.append(input_dict_4)
    
    # Input 5: Empty tensor
    input_tensor_5 = np.array([], dtype=np.int32)
    input_dict_5 = {
        "input": [tf.constant(input_tensor_5)],
        "name": "test_5"
    }
    list_of_inputs.append(input_dict_5)
    
    # Input 6: Mixed tensor
    input_tensor_6 = np.array([1, -2, 3], dtype=np.float32)
    input_dict_6 = {
        "input": [tf.constant(input_tensor_6)],
        "name": "test_6"
    }
    list_of_inputs.append(input_dict_6)
    
    # Input 7: 1D tensor with floating point values
    input_tensor_7 = np.array([1.5, 2.7, 3.14], dtype=np.float32)
    input_dict_7 = {
        "input": [tf.constant(input_tensor_7)],
        "name": "test_7"
    }
    list_of_inputs.append(input_dict_7)
    
    # Input 8: Multi-dimensional tensor
    input_tensor_8 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict_8 = {
        "input": [tf.constant(input_tensor_8)],
        "name": "test_8"
    }
    list_of_inputs.append(input_dict_8)
    
    # Input 9: Large tensor
    input_tensor_9 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=np.float64)
    input_dict_9 = {
        "input": [tf.constant(input_tensor_9)],
        "name": "test_9"
    }
    list_of_inputs.append(input_dict_9)
    
    # Input 10: Tensor with zero values
    input_tensor_10 = np.array([0, 0, 0], dtype=np.int32)
    input_dict_10 = {
        "input": [tf.constant(input_tensor_10)],
        "name": "test_10"
    }
    list_of_inputs.append(input_dict_10)

    return list_of_inputs

generated_inputs["tf.identity_n"] = tf_identity_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.identity_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.identity_n'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.identity_n', generated_inputs['tf.identity_n'], lib="tf", suffix=0)
