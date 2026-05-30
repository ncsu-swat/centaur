
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmax_inputs():
    list_of_inputs = []

    # Input 1: Float32 array with NaNs, x1 is True
    x1 = True
    x2 = np.array([1.5, -2.3, np.nan, np.inf], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 2: Int32 2D array, x1 is False
    x1 = False
    x2 = np.array([[1, -5], [10, 0]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 3: Float64 3D array, x1 is True
    x1 = True
    x2 = np.random.randn(2, 3, 2).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 4: Boolean array, x1 is False
    x1 = False
    x2 = np.array([True, False, True], dtype=np.bool_)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 5: Int64 1D array with negative values, x1 is True
    x1 = True
    x2 = np.array([-10, -20, 0, 50], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 6: Float16 array with infinity, x1 is False
    x1 = False
    x2 = np.array([[0.1, -0.2], [np.nan, -np.inf]], dtype=np.float16)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 7: Float32 with mixed special values, x1 is True
    x1 = True
    x2 = np.array([[-np.inf, np.inf], [np.nan, 0.0]], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 8: Int32 0-dimensional array, x1 is False
    x1 = False
    x2 = np.array(42, dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 9: Large float32 3D array, x1 is True
    x1 = True
    x2 = np.random.randn(1, 5, 5).astype(np.float32)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    # Input 10: Float64 1D array of NaNs, x1 is False
    x1 = False
    x2 = np.array([np.nan, np.nan], dtype=np.float64)
    list_of_inputs.append({"x1": x1, "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.fmax_7"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_7'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_7'], lib="jax", suffix=7)
