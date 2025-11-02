
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_squared_difference_inputs():
    list_of_inputs = []
    
    # Input 1: Basic tensor with positive values
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.float32)
    name = "input1"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Tensor with negative values
    x = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.float64)
    y = np.array([[2, 4, -6], [5, 7, 9]], dtype=np.float64)
    name = "input2"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Scalar tensors
    x = np.array(5.0, dtype=np.float32)
    y = np.array(3.0, dtype=np.float32)
    name = "input3"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 1D arrays
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([2, 3, 4, 5], dtype=np.int32)
    name = "input4"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Complex numbers
    x = np.array([1+2j, 3+4j], dtype=np.complex64)
    y = np.array([2+3j, 4+5j], dtype=np.complex64)
    name = "input5"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Broadcasting tensors (different shapes)
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[2, 3], [4, 5]], dtype=np.float32)
    name = "input6"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Zero tensors
    x = np.array([[0, 1], [2, 3]], dtype=np.float32)
    y = np.array([[0, 1], [2, 3]], dtype=np.float32)
    name = "input7"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Large difference (float64)
    x = np.array([[1000, 2000], [3000, 4000]], dtype=np.float64)
    y = np.array([[100, 200], [300, 400]], dtype=np.float64)
    name = "input8"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Large numbers (int64)
    x = np.array([[100, 200], [300, 400]], dtype=np.int64)
    y = np.array([[50, 100], [150, 200]], dtype=np.int64)
    name = "input9"
    
    input_dict = {
        "x": x,
        "y": y,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Mixed types (float32 and float64) - Removed to avoid error
    # x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # y = np.array([[2, 3], [4, 5]], dtype=np.float64)
    # name = "input10"
    
    # input_dict = {
    #     "x": x,
    #     "y": y,
    #     "name": name
    # }
    # list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.squared_difference"] = tf_squared_difference_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.squared_difference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.squared_difference'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.squared_difference', generated_inputs['tf.math.squared_difference'], lib="tf", suffix=0)
