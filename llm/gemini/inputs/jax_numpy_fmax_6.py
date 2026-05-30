
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, True
    x1 = np.array([1.5, -2.3, 0.0, -0.5], dtype=np.float32)
    x2 = True
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 2: 2D int32 array, False
    x1 = np.array([[2, -1], [0, 5]], dtype=np.int32)
    x2 = False
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 3: float64 array with NaN and Inf, True
    x1 = np.array([np.nan, np.inf, -np.inf, 3.14], dtype=np.float64)
    x2 = True
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 4: 3D float32 array, False
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = False
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 5: 1D float16 array, True
    x1 = np.array([-15.0, 0.5, 12.0], dtype=np.float16)
    x2 = True
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 6: 1D int64 array, False
    x1 = np.array([-100, 200, -300], dtype=np.int64)
    x2 = False
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 7: 4D float32 array, True
    x1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    x2 = True
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 8: 0D array, False
    x1 = np.array(-5.5, dtype=np.float32)
    x2 = False
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 9: Large values float64, True
    x1 = np.array([1e9, -1e9], dtype=np.float64)
    x2 = True
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    # Input 10: 2D float32 array with multiple NaNs, False
    x1 = np.array([[np.nan, 2.5], [-3.5, np.nan]], dtype=np.float32)
    x2 = False
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.fmax_6"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_6'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_6'], lib="jax", suffix=6)
