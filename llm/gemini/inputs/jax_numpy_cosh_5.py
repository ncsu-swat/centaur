
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cosh_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and zero values
    x = np.array([0.0, 1.0, -1.0, 2.5, -2.5], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array
    x = np.array([[1.0, -2.0, 0.5], [-1.5, 3.0, 0.0]], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32 array
    x = np.array([-3, -1, 0, 1, 4], dtype=np.int32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D single-element float32 array
    x = np.array([0.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex64 array
    x = np.array(
        [1.0 + 2.0j, -0.5j, 0.0 + 0.0j, -1.5 - 1.5j], dtype=np.complex64
    )
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex128 array
    x = np.array(
        [[1.0 - 1.0j, 2.0 + 0.5j], [-0.5 + 3.0j, 0.0 - 2.0j]],
        dtype=np.complex128,
    )
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty 1D array
    x = np.array([], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array
    x = np.zeros((2, 2, 2, 2), dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D boolean array
    x = np.array([True, False, True, False], dtype=np.bool_)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.cosh_5"] = cosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cosh_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cosh_5'.")


check_valid('jax.numpy.cosh', generated_inputs['jax.numpy.cosh_5'], lib="jax", suffix=5)
