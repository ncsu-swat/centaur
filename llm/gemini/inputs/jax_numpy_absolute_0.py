
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def absolute_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([-1.2, 0.0, 3.4, -5.6], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array with positive and negative values
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D int32 array
    x = np.random.randint(-100, 100, size=(2, 3, 3)).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Complex64 array
    x = np.array([3 + 4j, -5 - 12j, 0 + 0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Complex128 array
    real = np.random.randn(2, 2)
    imag = np.random.randn(2, 2)
    x = (real + 1j * imag).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D array (scalar)
    x = np.array(-42.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: int16 array with negative values
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: float32 array with special values (inf, nan)
    x = np.array([-np.inf, np.inf, np.nan, -0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D int64 array
    x = np.random.randint(-1000, 1000, size=(100,)).astype(np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.absolute"] = absolute_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.absolute' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.absolute'.")


check_valid('jax.numpy.absolute', generated_inputs['jax.numpy.absolute'], lib="jax", suffix=0)
