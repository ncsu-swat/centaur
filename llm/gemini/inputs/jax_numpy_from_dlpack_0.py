
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def from_dlpack_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, copy=True
    x = np.array([1.0, -2.5, 3.2, -4.8], dtype=np.float32)
    input_dict = {"x": x, "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array with negative values, copy=False
    x = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    input_dict = {"x": x, "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, copy=True
    x = np.ones((2, 3, 4), dtype=np.float64)
    input_dict = {"x": x, "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar float32 array, copy=True
    x = np.array(-5.5, dtype=np.float32)
    input_dict = {"x": x, "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D int64 array, copy=False
    x = np.arange(24, dtype=np.int64).reshape((2, 2, 2, 3))
    input_dict = {"x": x, "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D bool array, copy=True
    x = np.array([True, False, True, True], dtype=np.bool_)
    input_dict = {"x": x, "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 array, copy=False
    x = np.array([[0.5, 128.2], [255.1, 64.0]], dtype=np.float32)
    input_dict = {"x": x, "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D complex64 array, copy=True
    x = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64)
    input_dict = {"x": x, "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array with negatives, copy=False
    x = np.array([[-1.5, 2.5], [0.0, -3.5]], dtype=np.float32)
    input_dict = {"x": x, "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int32 array, copy=True
    x = np.array([-32768, 0, 32767], dtype=np.int32)
    input_dict = {"x": x, "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D int64 array, copy=False
    x = np.zeros((2, 3, 2), dtype=np.int64)
    input_dict = {"x": x, "copy": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.from_dlpack"] = from_dlpack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.from_dlpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.from_dlpack'.")


check_valid('jax.numpy.from_dlpack', generated_inputs['jax.numpy.from_dlpack'], lib="jax", suffix=0)
