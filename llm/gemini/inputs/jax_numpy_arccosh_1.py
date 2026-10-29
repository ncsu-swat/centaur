
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arccosh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 with values >= 1.0
    x = np.array([1.0, 1.5, 2.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array of float64
    x = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array with values including < 1 (producing NaNs for real inputs)
    x = np.array([-5.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D scalar array
    x = np.array(3.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array of int32
    x = np.arange(1, 13, dtype=np.int32).reshape((2, 2, 3))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D complex64 array
    x = np.array([1.0 + 2.0j, -5.0 + 0.0j, 0.0 + 1.0j, 2.0 - 3.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D complex128 array
    x = np.array([[-2.0 + 1.0j, 3.0 - 4.0j], [0.5 + 0.5j, 1.0 + 0.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D float32 array
    x = np.linspace(1.0, 50.0, 24, dtype=np.float32).reshape((2, 1, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D array containing infinities and large values
    x = np.array([1.0, 1e8, np.inf, 1e20], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 2D float16 array
    x = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.arccosh_1"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccosh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccosh_1'.")


check_valid('jax.numpy.arccosh', generated_inputs['jax.numpy.arccosh_1'], lib="jax", suffix=1)
