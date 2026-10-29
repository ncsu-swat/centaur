
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def asinh_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive, negative, zero float32
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array float64
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array int32
    x = np.array([[[-10, 0], [2, 5]], [[-1, 3], [7, -4]]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D array float16
    x = np.array([-2.5, -0.5, 0.5, 2.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array float32
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D complex64 array
    x = np.array(
        [[1.0 + 2.0j, -2.0 + 1.0j], [0.0 - 1.0j, 3.0 + 0.0j]],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D boolean array
    x = np.array([True, False, True, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 1D array with extreme values float64
    x = np.array([-1e12, -1e6, 0.0, 1e6, 1e12], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D int64 array
    x = np.arange(-6, 6, dtype=np.int64).reshape((3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D empty array float32
    x = np.empty((2, 0, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.asinh_1"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asinh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asinh_1'.")


check_valid('jax.numpy.asinh', generated_inputs['jax.numpy.asinh_1'], lib="jax", suffix=1)
