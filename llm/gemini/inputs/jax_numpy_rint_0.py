
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rint_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array with half-way values
    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D float64 array with normal floats
    x = np.array([-10.2, -5.8, 0.0, 1.2, 3.7, 100.5], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D float32 array
    x = np.random.uniform(-10, 10, size=(3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float32 array
    x = np.random.uniform(-5, 5, size=(2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D int32 array (integers are also valid tensors for rint)
    x = np.array([-5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0-dimensional scalar array
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Complex64 array
    x = np.array([-2.5+3.5j, 4.5-0.5j, 1.1+2.9j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Complex128 array with half-way values on real and imaginary parts
    x = np.array([0.5 + 0.5j, -1.5 - 2.5j], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D float32 array
    x = np.random.uniform(-100, 100, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Array containing special float values (nan, inf)
    x = np.array([-np.inf, np.nan, np.inf, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Array with large float64 values
    x = np.array([1e10 + 0.5, -2e15 - 0.5], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.rint"] = rint_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.rint' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.rint'.")


check_valid('jax.numpy.rint', generated_inputs['jax.numpy.rint'], lib="jax", suffix=0)
