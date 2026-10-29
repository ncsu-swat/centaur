
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def array_equiv_inputs():
    list_of_inputs = []

    # Input 1: 1D identical integer arrays
    a1 = np.array([1, 2, 3], dtype=np.int32)
    a2 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 2: 1D different integer arrays of same shape
    a1 = np.array([1, 2, 3], dtype=np.int64)
    a2 = np.array([1, 2, 4], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 3: 2D and 1D broadcastable equal arrays
    a1 = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
    a2 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 4: 0D scalar array and multi-dimensional array
    a1 = np.array(5.0, dtype=np.float32)
    a2 = np.full((3, 4), 5.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 5: Floating point arrays with negative values
    a1 = np.array([-1.5, 0.0, 2.75], dtype=np.float64)
    a2 = np.array([-1.5, 0.0, 2.75], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 6: Boolean arrays
    a1 = np.array([[True, False], [False, True]], dtype=np.bool_)
    a2 = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 7: Complex numbers
    a1 = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64)
    a2 = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 8: 3D arrays broadcasting with (1, 1, 3) and (2, 4, 3)
    a1 = np.zeros((1, 1, 3), dtype=np.float32)
    a2 = np.zeros((2, 4, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 9: 2D arrays broadcastable along axis 0
    a1 = np.array([[1], [2], [3]], dtype=np.int32)
    a2 = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 10: Empty arrays with matching shapes
    a1 = np.empty((0, 5), dtype=np.float32)
    a2 = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 11: 4D arrays with different broadcastable shapes
    a1 = np.ones((2, 1, 3, 1), dtype=np.int16)
    a2 = np.ones((1, 4, 1, 5), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    # Input 12: Arrays containing NaNs
    a1 = np.array([np.nan, 1.0], dtype=np.float32)
    a2 = np.array([np.nan, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a1": a1, "a2": a2}))

    return list_of_inputs


generated_inputs["jax.numpy.array_equiv"] = array_equiv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_equiv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_equiv'.")


check_valid('jax.numpy.array_equiv', generated_inputs['jax.numpy.array_equiv'], lib="jax", suffix=0)
