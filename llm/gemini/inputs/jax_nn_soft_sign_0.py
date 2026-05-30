
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def soft_sign_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, positive values
    x = np.array([0.5, 1.0, 2.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array, float32, negative values
    x = np.array([-0.5, -1.0, -2.0, -10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array, float32, mixed positive/negative/zeros
    x = np.array([[0.0, -1.5, 2.5], [-3.0, 4.0, 0.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array, float64, random values
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0-D array (scalar), float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D array, float32, simulating image batch
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D array, float32, extreme large and small values
    x = np.array([-1e6, -1e-6, 0.0, 1e-6, 1e6], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D array, float64, uniform values
    x = np.random.uniform(-5.0, 5.0, size=(5, 5)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array, float32, containing only zeros
    x = np.zeros((10,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D array, float32, complex dimensions
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.soft_sign"] = soft_sign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.soft_sign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.soft_sign'.")


check_valid('jax.nn.soft_sign', generated_inputs['jax.nn.soft_sign'], lib="jax", suffix=0)
