
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arcsinh_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive values
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array with negative and positive values
    x = np.array([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float64 array with wide range of values
    x = np.random.uniform(-100.0, 100.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D int32 array (should be promoted to inexact in JAX)
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D complex64 array
    x = np.array([4.0 - 3.0j, 2.0j, -1.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D complex128 array
    x = np.array([[1.0 + 2.0j, -3.0j], [4.0, -5.0 - 6.0j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 0D array (scalar equivalent)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Float32 array containing infinity and nan
    x = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: High-dimensional (4D) float32 array
    x = np.random.normal(size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D int64 array
    x = np.array([[100, -200], [300, -400]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.arcsinh_1"] = arcsinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arcsinh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arcsinh_1'.")


check_valid('jax.numpy.arcsinh', generated_inputs['jax.numpy.arcsinh_1'], lib="jax", suffix=1)
