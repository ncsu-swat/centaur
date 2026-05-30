
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def abs_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers
    list_of_inputs.append({"x": np.array([-1, 2, -3, 0, 4], dtype=np.int32)})

    # Input 2: 1D array of floats
    list_of_inputs.append({"x": np.array([-1.5, 2.7, -3.14, 0.0, 5.5], dtype=np.float32)})

    # Input 3: 2D array of integers
    list_of_inputs.append({"x": np.array([[-1, -2, -3], [4, 5, 6]], dtype=np.int64)})

    # Input 4: 2D array of floats
    list_of_inputs.append({"x": np.array([[-1.1, 2.2], [-3.3, 4.4]], dtype=np.float64)})

    # Input 5: 1D array of complex numbers
    list_of_inputs.append({"x": np.array([1 + 1j, -2 - 2j, 3j, -4j], dtype=np.complex64)})

    # Input 6: 1D array of single float (replacing 0D scalar array)
    list_of_inputs.append({"x": np.array([-9.9], dtype=np.float32)})

    # Input 7: 1D Boolean array
    list_of_inputs.append({"x": np.array([True, False, True], dtype=bool)})

    # Input 8: 3D random array of floats
    list_of_inputs.append({"x": np.random.randn(2, 3, 4).astype(np.float32)})

    # Input 9: 1D array with infinity and nan values
    list_of_inputs.append({"x": np.array([float('-inf'), float('inf'), float('nan')], dtype=np.float32)})

    # Input 10: 1D array of int8 containing negative boundaries
    list_of_inputs.append({"x": np.array([-128, -1, 0, 127], dtype=np.int8)})

    return list_of_inputs

generated_inputs["jax.numpy.abs_5"] = abs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.abs_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.abs_5'.")


check_valid('jax.numpy.abs', generated_inputs['jax.numpy.abs_5'], lib="jax", suffix=5)
