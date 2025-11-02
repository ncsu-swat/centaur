
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_crelu_inputs():
    list_of_inputs = []
    
    # Input 1: 2D tensor with positive and negative values
    features = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)
    axis = -1
    name = "test1"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 3D tensor with positive and negative values
    features = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.float32)
    axis = -1
    name = "test2"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 1D tensor with positive and negative values
    features = np.array([1, -2, 3, -4, 5], dtype=np.float32)
    axis = -1
    name = "test3"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4D tensor with positive and negative values
    features = np.array([[[[1, -2], [3, -4]], [[5, -6], [7, -8]]], [[[-9, 10], [-11, 12]], [[13, -14], [-15, 16]]]], dtype=np.float32)
    axis = -1
    name = "test4"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D tensor with all positive values
    features = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = -1
    name = "test5"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 2D tensor with all negative values
    features = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float32)
    axis = -1
    name = "test6"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D tensor with mixed values, different axis
    features = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float32)
    axis = 0
    name = "test7"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor with mixed values, different axis
    features = np.array([1, -2, 3, -4, 5], dtype=np.float32)
    axis = 0
    name = "test8"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 3D tensor with positive and negative values, different axis
    features = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.float32)
    axis = 0
    name = "test9"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 4D tensor with all positive values, different axis
    features = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], dtype=np.float32)
    axis = 1
    name = "test10"
    
    input_dict = {
        "features": features,
        "axis": axis,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.nn.crelu"] = generate_crelu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.crelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.crelu'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.crelu', generated_inputs['tf.nn.crelu'], lib="tf", suffix=0)
