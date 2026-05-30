
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ldexp_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays of float32 and int32
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = np.array([0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 2: 1D arrays with float64 and int64
    x1 = np.array([-1.5, 0.5, 2.75], dtype=np.float64)
    x2 = np.array([-1, 2, -3], dtype=np.int64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 3: 2D arrays, same shape
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 4: 0D arrays (scalars wrapped as arrays)
    x1 = np.array(5.5, dtype=np.float32)
    x2 = np.array(3, dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 5: Broadcasting x2 (1D) to x1 (2D)
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x2 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 6: Broadcasting x1 (1D) to x2 (2D)
    x1 = np.array([0.5, 1.5], dtype=np.float32)
    x2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 7: Float16 and Int16 types
    x1 = np.array([1.0, -2.0, 3.0], dtype=np.float16)
    x2 = np.array([5, -5, 0], dtype=np.int16)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 8: 3D arrays
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 9: Large values and negative values for float/int
    x1 = np.array([1e-5, -1e5, 0.0], dtype=np.float32)
    x2 = np.array([10, -10, 5], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 10: One element arrays (1D of size 1)
    x1 = np.array([3.14], dtype=np.float64)
    x2 = np.array([-2], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.ldexp_1"] = ldexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ldexp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ldexp_1'.")


check_valid('jax.numpy.ldexp', generated_inputs['jax.numpy.ldexp_1'], lib="jax", suffix=1)
