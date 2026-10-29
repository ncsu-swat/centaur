
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isrealobj_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32 array
    x = np.array(3.14, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D integer array with negative values
    x = np.array([-5, 0, 10, -20], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float64 array
    x = np.array([[-1.5, 2.5], [0.0, -4.2]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D complex64 array
    x = np.array([1.0 + 2.0j, -3.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D complex128 array
    x = np.array([[0.0 + 1.0j, 2.0 - 3.0j], [-1.0 + 0.0j, 4.0 + 5.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Boolean array
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D uint8 array
    x = np.zeros((2, 3, 4), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty array
    x = np.empty((0, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D float16 array
    x = np.ones((2, 2, 2, 2), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 0D complex array
    x = np.array(1.0 + 0.0j, dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D int64 array
    x = np.array([1000000000000, -2000000000000], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.isrealobj_1"] = isrealobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isrealobj_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isrealobj_1'.")


check_valid('jax.numpy.isrealobj', generated_inputs['jax.numpy.isrealobj_1'], lib="jax", suffix=1)
