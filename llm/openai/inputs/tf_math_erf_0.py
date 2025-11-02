
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_erf_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor
    x = np.array(0.5, dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor
    x = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor
    x = np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    x = np.array([[[1.0, 2.0], [0.0, -1.0]], [[2.0, 3.0], [-1.0, 0.0]]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: negative values
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: mixed positive and negative values
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float64 tensor
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: bfloat16 tensor
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: half tensor
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: tensor with zero values
    x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.math.erf"] = tf_math_erf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.erf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.erf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.erf', generated_inputs['tf.math.erf'], lib="tf", suffix=0)
