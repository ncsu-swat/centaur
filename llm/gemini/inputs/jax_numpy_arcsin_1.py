
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arcsin_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 values within [-1, 1]
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 0D scalar tensor
    x = np.array(0.7071, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array of float64 values
    x = np.array([[-0.9, -0.1], [0.1, 0.9]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array of float16 values
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float16).reshape(2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D array with values outside [-1, 1] (returns nan)
    x = np.array([-2.0, -1.5, 0.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Complex64 1D array
    x = np.array([1.0 + 2.0j, -0.5 + 0.5j, 3.0 - 4.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Complex128 2D array
    x = np.array(
        [[0.0 + 1.0j, -1.0 + 0.0j], [2.0 - 1.0j, 0.5 + 0.5j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Integer array (promoted to inexact dtype by jax)
    x = np.array([-1, 0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty array
    x = np.empty((0, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 4D array
    x = np.random.uniform(-1.0, 1.0, size=(2, 3, 4, 5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Boolean array (promoted to inexact)
    x = np.array([True, False, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.arcsin_1"] = arcsin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arcsin_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arcsin_1'.")


check_valid('jax.numpy.arcsin', generated_inputs['jax.numpy.arcsin_1'], lib="jax", suffix=1)
