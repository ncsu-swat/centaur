
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_unicode_script_inputs():
    list_of_inputs = []
    
    # Input 1: Simple 1D array
    input_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {
        "input": input_array,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 2D array
    input_array_2d = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        "input": input_array_2d,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Single element
    input_array_single = np.array([100], dtype=np.int32)
    input_dict = {
        "input": input_array_single,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Negative values
    input_array_neg = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {
        "input": input_array_neg,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Mixed values including negative
    input_array_mix = np.array([1, -2, 3], dtype=np.int32)
    input_dict = {
        "input": input_array_mix,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Large values
    input_array_large = np.array([1000, 2000, 3000], dtype=np.int32)
    input_dict = {
        "input": input_array_large,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D array
    input_array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {
        "input": input_array_3d,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Single element in 1D array
    input_array_single_dim = np.array([0], dtype=np.int32)
    input_dict = {
        "input": input_array_single_dim,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Empty array
    input_array_empty = np.array([], dtype=np.int32)
    input_dict = {
        "input": input_array_empty,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Complex mix of values
    input_array_complex = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    input_dict = {
        "input": input_array_complex,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.UnicodeScript"] = tf_unicode_script_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.UnicodeScript' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.UnicodeScript'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.UnicodeScript', generated_inputs['tf.raw_ops.UnicodeScript'], lib="tf", suffix=0)
