
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def generate_log1p_inputs():
    list_of_inputs = []
    
    # Input 1: scalar tensor
    x = np.array(0.5, dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: 1D tensor with negative values
    x = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: 2D tensor
    x = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 3D tensor
    x = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 1D tensor with complex numbers
    x = np.array([1+1j, 2+2j], dtype=np.complex64)
    input_dict = {
        "x": x,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 1D tensor with float64
    x = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: scalar tensor with float32
    x = np.array(1.5, dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: 1D tensor with half precision
    x = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    input_dict = {
        "x": x,
        "name": "test8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: 1D tensor with bfloat16
    x = np.array([0.5, 1.0, 1.5], dtype=np.float16)
    input_dict = {
        "x": x,
        "name": "test9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: 1D tensor with negative values
    x = np.array([-0.9, -0.5, -0.1], dtype=np.float32)
    input_dict = {
        "x": x,
        "name": "test10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.log1p"] = generate_log1p_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.log1p'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.log1p', generated_inputs['tf.math.log1p'], lib="tf", suffix=0)
