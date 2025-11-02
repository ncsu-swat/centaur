
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_cos_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 200, 10000, np.inf], dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with mixed values
    x = np.array([-1.5, 0.0, 1.5, 3.14], dtype=np.float64)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: complex64 tensor
    x = np.array([1+2j, 3+4j], dtype=np.complex64)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: complex128 tensor
    x = np.array([1+2j, 3+4j], dtype=np.complex128)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: scalar tensor
    x = np.array(1.0, dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 1D tensor with negative values
    x = np.array([-5, -2.5, 0, 2.5, 5], dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D tensor with negative values
    x = np.array([[-1, -2], [3, 4]], dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 3D tensor with negative values
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32 tensor with nan values
    x = np.array([np.nan, -np.inf, np.inf], dtype=np.float32)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: float64 tensor with zero values
    x = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    input_dict = {
        "x": tf.constant(x),
        "name": "cos_test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.cos"] = generate_cos_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.cos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.cos'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.cos', generated_inputs['tf.math.cos'], lib="tf", suffix=0)
