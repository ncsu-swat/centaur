
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def signbit_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with mixed values
    x = np.array([-1.5, -0.0, 0.0, 1.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D boolean array
    x = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D int32 array with negative and positive values
    x = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D float64 array with signed zeros
    x = np.array([-0.0, 0.0, -0.0, 0.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Float32 array with special values (NaN, Inf)
    x = np.array([np.nan, -np.nan, np.inf, -np.inf], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 3D int16 array
    x = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 0D (scalar) float32 array
    x = np.array(-5.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Large 2D float32 array
    x = np.random.randn(10, 10).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D float64 array
    x = (np.ones((2, 2, 2, 2), dtype=np.float64) * -1.0)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D int64 array with boundary values
    x = np.array([-9223372036854775808, 9223372036854775807, 0], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D uint8 array (all values should return False since they are non-negative)
    x = np.array([0, 1, 128, 255], dtype=np.uint8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.signbit"] = signbit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.signbit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.signbit'.")


check_valid('jax.numpy.signbit', generated_inputs['jax.numpy.signbit'], lib="jax", suffix=0)
