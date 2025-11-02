
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def sinh_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-np.inf, -9, -0.5, 1, 1.2, 2, 10, np.inf], dtype=np.float32)
    input_dict = {
        "name": "test1",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 2: float64 tensor with mixed values
    x = np.array([-np.inf, -9.5, -0.5, 1.5, 1.2, 2.7, 10.3, np.inf], dtype=np.float64)
    input_dict = {
        "name": "test2",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 3: half tensor with positive values
    x = np.array([0.5, 1.2, 2.7, 10.3], dtype=np.float16)
    input_dict = {
        "name": "test3",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 4: bfloat16 tensor with negative values
    x = np.array([-1.5, -2.7, -10.3], dtype=np.float32)
    input_dict = {
        "name": "test4",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 5: complex64 tensor with real part
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict = {
        "name": "test5",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 6: complex128 tensor with real part
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    input_dict = {
        "name": "test6",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 7: scalar tensor (0D)
    x = np.array(5.5, dtype=np.float32)
    input_dict = {
        "name": "test7",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 8: 1D tensor with multiple negative values
    x = np.array([-1.5, -2.7, -10.3, -100], dtype=np.float32)
    input_dict = {
        "name": "test8",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 9: 3D tensor with multiple dimensions
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict = {
        "name": "test9",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    # Input 10: tensor with zero values
    x = np.array([0.0, -0.5, 1.5], dtype=np.float32)
    input_dict = {
        "name": "test10",
        "x": x
    }
    list_of_inputs.append(input_dict)
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Sinh"] = sinh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Sinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sinh'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Sinh', generated_inputs['tf.raw_ops.Sinh'], lib="tf", suffix=0)
