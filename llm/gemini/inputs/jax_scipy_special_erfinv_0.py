
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def erfinv_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({"x": np.array([-0.9, -0.5, 0.0, 0.5, 0.9], dtype=np.float32)})

    # Input 2
    list_of_inputs.append({"x": np.array(0.3, dtype=np.float32)})

    # Input 3
    list_of_inputs.append({"x": np.array([[-0.2, 0.4], [0.1, -0.8]], dtype=np.float32)})

    # Input 4
    list_of_inputs.append({"x": np.array([[[0.1, -0.1], [0.2, -0.2]], [[0.3, -0.3], [0.4, -0.4]]], dtype=np.float64)})

    # Input 5
    list_of_inputs.append({"x": np.array([-0.1, -0.01, 0.0, 0.01, 0.1], dtype=np.float64)})

    # Input 6
    list_of_inputs.append({"x": np.array([0.95, 0.99, 0.999], dtype=np.float32)})

    # Input 7
    list_of_inputs.append({"x": np.array([-0.95, -0.99, -0.999], dtype=np.float32)})

    # Input 8
    list_of_inputs.append({"x": np.array([[[[0.05, -0.05]]]], dtype=np.float32)})

    # Input 9
    list_of_inputs.append({"x": np.zeros((5,), dtype=np.float32)})

    # Input 10
    list_of_inputs.append({"x": np.array([[-0.7, 0.7], [-0.2, 0.2]], dtype=np.float64)})

    return list_of_inputs

generated_inputs["jax.scipy.special.erfinv"] = erfinv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.erfinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.erfinv'.")


check_valid('jax.scipy.special.erfinv', generated_inputs['jax.scipy.special.erfinv'], lib="jax", suffix=0)
