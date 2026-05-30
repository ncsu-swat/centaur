
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sqrt_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive floats (float32)
    x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array of positive floats (float32)
    x = np.array([[2.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D array of positive floats (float64)
    x = np.random.uniform(1.0, 10.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D array containing negative floats (to test nan output for real negatives)
    x = np.array([-1.0, -4.0, 9.0, -16.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D complex numbers
    x = np.array([-8-6j, 1j, 4+0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D array (scalar equivalent)
    x = np.array(25.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Positive integers (int32)
    x = np.array([1, 4, 9, 16, 25], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Float16 array for half-precision
    x = np.array([0.5, 1.5, 2.5], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Array with zero and infinity values
    x = np.array([0.0, np.inf], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 4D high-dimensional array
    x = np.random.exponential(scale=2.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Complex128 double-precision complex array
    x = np.array([-2+3j, -4-5j], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.sqrt_1"] = sqrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sqrt_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sqrt_1'.")


check_valid('jax.numpy.sqrt', generated_inputs['jax.numpy.sqrt_1'], lib="jax", suffix=1)
