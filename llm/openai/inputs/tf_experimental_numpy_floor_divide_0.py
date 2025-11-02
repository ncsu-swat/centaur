
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_floor_divide_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    x2 = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    x2 = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3, valid
    x1 = np.array([1, 2, 3], dtype=np.float32)
    x2 = np.array([2, 4, -6], dtype=np.float32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4, valid
    x1 = np.array([[1.5, 2.7], [3.9, 4.1]], dtype=np.float64)
    x2 = np.array([[2.3, 4.5], [6.7, 8.9]], dtype=np.float64)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5, valid
    x1 = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    x2 = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    x1 = np.array([[-1.5, -2.7], [-3.9, -4.1]], dtype=np.float64)
    x2 = np.array([[2.3, 4.5], [6.7, 8.9]], dtype=np.float64)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, valid
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    x2 = np.array([[2, 3], [4, 5]], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8, valid
    x1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    x2 = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid
    x1 = np.array([1, 2, 3], dtype=np.int32)
    x2 = np.array([2, 4, -6], dtype=np.int32)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid
    x1 = np.array([[1.5, 2.7], [3.9, 4.1]], dtype=np.float64)
    x2 = np.array([[2.3, 4.5], [6.7, 8.9]], dtype=np.float64)
    
    input_dict = {
        "x1": x1,
        "x2": x2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.experimental.numpy.floor_divide"] = tf_floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.floor_divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.floor_divide'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.floor_divide', generated_inputs['tf.experimental.numpy.floor_divide'], lib="tf", suffix=0)
