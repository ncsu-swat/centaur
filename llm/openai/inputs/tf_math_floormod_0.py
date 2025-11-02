
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def floormod_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    name = "test1"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int64)
    name = "test2"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 4.0, -6.0], dtype=np.float32)
    name = "test3"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([2.0, 4.0, -6.0], dtype=np.float64)
    name = "test4"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid
    x = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    y = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.int32)
    name = "test5"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid
    x = np.array([[1, 2], [3, 4]], dtype=np.int16)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int16)
    name = "test6"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid
    x = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float64)
    y = np.array([[2.0, 4.0], [-6.0, 8.0]], dtype=np.float64)
    name = "test7"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid
    x = np.array([1, 2, 3], dtype=np.int8)
    y = np.array([2, 4, -6], dtype=np.int8)
    name = "test8"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9, valid
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    y = np.array([[2, 3], [4, 5]], dtype=np.uint8)
    name = "test9"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10, valid
    x = np.array([[-1, -2], [-3, -4]], dtype=np.uint8)
    y = np.array([[2, 4], [-6, 8]], dtype=np.uint8)
    name = "test10"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.math.floormod"] = floormod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.floormod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.floormod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.floormod', generated_inputs['tf.math.floormod'], lib="tf", suffix=0)
