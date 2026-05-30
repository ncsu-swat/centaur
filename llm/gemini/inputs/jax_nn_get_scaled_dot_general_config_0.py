
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_scaled_dot_general_config_inputs():
    list_of_inputs = []

    # Input 1, nvfp4 mode, scalar float32 scale
    input_dict = {
        "mode": "nvfp4",
        "global_scale": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, mxfp8 mode, scalar float32 scale
    input_dict = {
        "mode": "mxfp8",
        "global_scale": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, nvfp4 mode, 1D float32 scale
    input_dict = {
        "mode": "nvfp4",
        "global_scale": np.array([1.0, 2.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, mxfp8 mode, 1D float32 scale with multiple elements
    input_dict = {
        "mode": "mxfp8",
        "global_scale": np.array([0.5, 1.5, 2.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, nvfp4 mode, scalar float16 scale
    input_dict = {
        "mode": "nvfp4",
        "global_scale": np.array(2.0, dtype=np.float16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, mxfp8 mode, 2D float32 scale
    input_dict = {
        "mode": "mxfp8",
        "global_scale": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, nvfp4 mode, 3D float32 scale
    input_dict = {
        "mode": "nvfp4",
        "global_scale": np.ones((2, 2, 2), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, mxfp8 mode, high scale value
    input_dict = {
        "mode": "mxfp8",
        "global_scale": np.array(100.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, nvfp4 mode, very small scale value
    input_dict = {
        "mode": "nvfp4",
        "global_scale": np.array(1e-5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, mxfp8 mode, float64 scale
    input_dict = {
        "mode": "mxfp8",
        "global_scale": np.array(1.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.get_scaled_dot_general_config"] = get_scaled_dot_general_config_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.get_scaled_dot_general_config' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.get_scaled_dot_general_config'.")


check_valid('jax.nn.get_scaled_dot_general_config', generated_inputs['jax.nn.get_scaled_dot_general_config'], lib="jax", suffix=0)
