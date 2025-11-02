
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_log_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    input_dict = {
        'name': 'test1',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with negative values
    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float64)
    input_dict = {
        'name': 'test2',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: complex64 tensor with negative values
    x = np.array([-1.0+0j, -0.5+0j, 0.5+0j, 1.0+0j], dtype=np.complex64)
    input_dict = {
        'name': 'test3',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: complex128 tensor with negative values
    x = np.array([-1.0+0j, -0.5+0j, 0.5+0j, 1.0+0j], dtype=np.complex128)
    input_dict = {
        'name': 'test4',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: float32 tensor with zero values
    x = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    input_dict = {
        'name': 'test5',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float64 tensor with zero values
    x = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    input_dict = {
        'name': 'test6',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: complex64 tensor with zero values
    x = np.array([0.0+0j, 0.5+0j, 1.0+0j, 2.0+0j], dtype=np.complex64)
    input_dict = {
        'name': 'test7',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: complex128 tensor with zero values
    x = np.array([0.0+0j, 0.5+0j, 1.0+0j, 2.0+0j], dtype=np.complex128)
    input_dict = {
        'name': 'test8',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: half tensor with negative values
    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float16)
    input_dict = {
        'name': 'test9',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: bfloat16 tensor with negative values
    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    input_dict = {
        'name': 'test10',
        'x': x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Log"] = tf_raw_ops_log_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Log' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Log'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Log', generated_inputs['tf.raw_ops.Log'], lib="tf", suffix=0)
