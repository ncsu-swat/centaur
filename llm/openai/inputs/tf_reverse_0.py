
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_reverse_inputs():
    list_of_inputs = []
    
    # Input 1: 4D tensor with axis [3]
    tensor_1 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_1 = np.array([3], dtype=np.int32)
    input_dict_1 = {
        "tensor": tensor_1,
        "axis": axis_1,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    
    # Input 2: 4D tensor with axis [-1]
    tensor_2 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_2 = np.array([-1], dtype=np.int32)
    input_dict_2 = {
        "tensor": tensor_2,
        "axis": axis_2,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    
    # Input 3: 4D tensor with axis [1]
    tensor_3 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_3 = np.array([1], dtype=np.int32)
    input_dict_3 = {
        "tensor": tensor_3,
        "axis": axis_3,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: 4D tensor with axis [-3]
    tensor_4 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_4 = np.array([-3], dtype=np.int32)
    input_dict_4 = {
        "tensor": tensor_4,
        "axis": axis_4,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: 4D tensor with axis [2]
    tensor_5 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_5 = np.array([2], dtype=np.int32)
    input_dict_5 = {
        "tensor": tensor_5,
        "axis": axis_5,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: 4D tensor with axis [-2]
    tensor_6 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_6 = np.array([-2], dtype=np.int32)
    input_dict_6 = {
        "tensor": tensor_6,
        "axis": axis_6,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: 3D tensor with axis [0]
    tensor_7 = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], dtype=np.int32)
    axis_7 = np.array([0], dtype=np.int32)
    input_dict_7 = {
        "tensor": tensor_7,
        "axis": axis_7,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: 2D tensor with axis [1]
    tensor_8 = np.array([[0, 1], [2, 3]], dtype=np.int32)
    axis_8 = np.array([1], dtype=np.int32)
    input_dict_8 = {
        "tensor": tensor_8,
        "axis": axis_8,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: 4D tensor with axis [0, 1]
    tensor_9 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_9 = np.array([0, 1], dtype=np.int32)
    input_dict_9 = {
        "tensor": tensor_9,
        "axis": axis_9,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: 4D tensor with axis [1, 3]
    tensor_10 = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]], dtype=np.int32)
    axis_10 = np.array([1, 3], dtype=np.int32)
    input_dict_10 = {
        "tensor": tensor_10,
        "axis": axis_10,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["tf.reverse"] = tf_reverse_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.reverse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.reverse'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.reverse', generated_inputs['tf.reverse'], lib="tf", suffix=0)
