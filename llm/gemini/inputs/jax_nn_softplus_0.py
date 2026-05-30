
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def softplus_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 array with negative, positive, and zero values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: Standard 2D float32 array
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float32 array
    x = np.random.randn(2, 3, 5).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Scalar (0D array) float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: float64 array
    x = np.random.randn(5).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: float16 array
    x = np.random.randn(4, 2).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Large positive values (testing numerical stability)
    x = np.array([50.0, 100.0, 1000.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Large negative values (testing values near zero)
    x = np.array([-50.0, -100.0, -1000.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Array with all zeros
    x = np.zeros((2, 2), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Uniform random values in a larger range
    x = np.random.uniform(-10.0, 10.0, size=(10, 10)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.softplus"] = softplus_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.softplus'.")


check_valid('jax.nn.softplus', generated_inputs['jax.nn.softplus'], lib="jax", suffix=0)
