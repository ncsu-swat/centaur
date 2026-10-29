
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def atanh_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array, float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with values in (-1, 1), float32
    x = np.array([-0.8, -0.5, 0.0, 0.5, 0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array, float64
    x = np.array([[-0.9, 0.1], [0.3, -0.7]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array, float16
    x = np.linspace(-0.9, 0.9, 8, dtype=np.float16).reshape(2, 2, 2)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D complex64 array
    x = np.array(
        [0.5 + 0.5j, -0.2 - 0.3j, 1.5 + 0.0j], dtype=np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array with zeros and small values, float32
    x = np.zeros((2, 1, 3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D complex128 array
    x = np.array(
        [[-0.5 + 0.2j, 0.0 + 0.0j], [0.8 - 0.1j, -0.3 + 0.9j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D integer array
    x = np.array([0, 0, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D array close to boundary values, float32
    x = np.array([-0.999, -0.5, 0.0, 0.5, 0.999], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 5D array, float32
    x = (np.random.uniform(-0.8, 0.8, size=(1, 2, 2, 1, 3))).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 2D empty array, float32
    x = np.empty((0, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.atanh_1"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atanh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atanh_1'.")


check_valid('jax.numpy.atanh', generated_inputs['jax.numpy.atanh_1'], lib="jax", suffix=1)
