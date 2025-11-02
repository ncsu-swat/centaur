
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def generate_tan_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 200, 10000, np.inf], dtype=np.float32)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 200, 10000, np.inf], dtype=np.float64)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: complex64 tensor with real and imaginary parts
    x = np.array([complex(-0.5, 1.2), complex(1, 2), complex(3, 4)], dtype=np.complex64)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: complex128 tensor with real and imaginary parts
    x = np.array([complex(-0.5, 1.2), complex(1, 2), complex(3, 4)], dtype=np.complex128)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: half tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 200, 10000, np.inf], dtype=np.float16)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: float32 scalar tensor
    x = np.array(1.5, dtype=np.float32)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: float64 scalar tensor
    x = np.array(1.5, dtype=np.float64)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: half scalar tensor
    x = np.array(1.5, dtype=np.float16)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: complex64 scalar tensor
    x = np.array(complex(1.5, 2.5), dtype=np.complex64)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: complex128 scalar tensor
    x = np.array(complex(1.5, 2.5), dtype=np.complex128)
    input_dict = {
        "name": "tan",
        "x": tf.constant(x)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Tan"] = generate_tan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Tan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Tan'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Tan', generated_inputs['tf.raw_ops.Tan'], lib="tf", suffix=0)
