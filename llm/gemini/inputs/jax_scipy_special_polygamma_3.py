
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_special_polygamma_inputs():
    list_of_inputs = []

    # Input 1: Scalar int and float
    n = np.array(1, dtype=np.int32)
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"n": copy.deepcopy(n), "x": copy.deepcopy(x)})

    # Input 2: 1D arrays
    n = np.array([0, 1, 2], dtype=np.int32)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"n": copy.deepcopy(n), "x": copy.deepcopy(x)})

    # Input 3: 2D arrays, int64 and float64
    n = np.array([[1, 2], [3, 4]], dtype=np.int64)
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    list_of_inputs.append({"n": copy.deepcopy(n), "x": copy.deepcopy(x)})

    # Input 4: Scalar n, 1D x
    n = np.array(0, dtype=np.int32)
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    list_of_inputs.append({"n": copy.deepcopy(n), "x": copy.deepcopy(x)})

    # Input 5: 1D n, scalar x
    n = np.array([1, 2], dtype=np.int32)
    x = np.array(5.0, dtype=np.float64)
    list_of_inputs.append({"n": copy.deepcopy(n), "x": copy.deepcopy(x)})

    # Input 6: 3D arrays
    n = np.array([[[1]]], dtype=np.int32)
    x = np.array([[[10.0]]], dtype=np.float32)
    list_of_inputs.append({"n": copy.deepcopy(n), "x": copy.deepcopy(x)})

    # Input 7: Small positive x values
    n = np.array([0, 0, 1, 1], dtype=np.int32)
    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    list_of_inputs.append({"n": copy.deepcopy(n), "x": copy.deepcopy(x)})

    # Input 8: Large x values, float64
    n = np.array(3, dtype=np.int32)
    x = np.array([100.0, 200.0], dtype=np.float64)
    list_of_inputs.append({"n": copy.deepcopy(n), "x": copy.deepcopy(x)})

    # Input 9: High-dimensional arrays
    n = np.ones((2, 3, 2), dtype=np.int32)
    x = np.random.uniform(0.5, 5.0, size=(2, 3, 2)).astype(np.float32)
    list_of_inputs.append({"n": copy.deepcopy(n), "x": copy.deepcopy(x)})

    # Input 10: 1-element 1D arrays
    n = np.array([2], dtype=np.int64)
    x = np.array([1.1], dtype=np.float64)
    list_of_inputs.append({"n": copy.deepcopy(n), "x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.scipy.special.polygamma_3"] = jax_scipy_special_polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.polygamma_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.polygamma_3'.")


check_valid('jax.scipy.special.polygamma', generated_inputs['jax.scipy.special.polygamma_3'], lib="jax", suffix=3)
