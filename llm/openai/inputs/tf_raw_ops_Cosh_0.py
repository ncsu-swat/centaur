
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def generate_cosh_inputs():
    list_of_inputs = []
    
    # Input 1: float32 tensor with negative values
    x = np.array([-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: float64 tensor with mixed values
    x = np.array([-9.5, -0.5, 0.0, 1.5, 2.5], dtype=np.float64)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: half tensor with complex values
    x = np.array([1+1j, 2+2j, 0+0j], dtype=np.complex64)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: bfloat16 tensor with large values
    x = np.array([100, 200, 300], dtype=np.float32)  # Note: bfloat16 not directly supported, but using float32 as proxy
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: scalar tensor (1D)
    x = np.array([1.5], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: 1D tensor with negative values
    x = np.array([-1, -2, -3], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 2D tensor (not just one dimension)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: complex128 tensor with real values
    x = np.array([1.0j, 2.0j], dtype=np.complex128)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: float32 tensor with very small values
    x = np.array([0.001, 0.0001], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: tensor with zero values
    x = np.array([0, 0, 0], dtype=np.float32)
    input_dict = {"name": "Cosh", "x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Cosh"] = generate_cosh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Cosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cosh'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Cosh', generated_inputs['tf.raw_ops.Cosh'], lib="tf", suffix=0)
