
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def select_n_inputs():
    list_of_inputs = []

    # Input 1: 1D array cases, 1D int32 which (same shape), float32
    cases = np.array([1.0, -2.0, 3.5, -4.5], dtype=np.float32)
    which = np.zeros((4,), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"which": which, "cases": cases}))

    # Input 2: 2D array cases, scalar int32 which, int32
    cases = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int32)
    which = np.array(0, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"which": which, "cases": cases}))

    # Input 3: 2D array cases, 2D int32 which (same shape), float64
    cases = np.arange(12, dtype=np.float64).reshape(3, 4) - 5.0
    which = np.zeros((3, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"which": which, "cases": cases}))

    # Input 4: 1D array cases, scalar bool which, bool
    cases = np.array([True, False, True, False], dtype=np.bool_)
    which = np.array(False, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"which": which, "cases": cases}))

    # Input 5: 3D array cases, 3D int32 which (same shape), complex64
    cases = np.ones((2, 3, 2), dtype=np.complex64) * (1.5 - 2.5j)
    which = np.zeros((2, 3, 2), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"which": which, "cases": cases}))

    # Input 6: 0D scalar cases, scalar int32 which, float32
    cases = np.array(3.14, dtype=np.float32)
    which = np.array(0, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"which": which, "cases": cases}))

    # Input 7: 4D array cases, 4D int32 which (same shape), int16
    cases = np.arange(24, dtype=np.int16).reshape(2, 2, 2, 3) - 10
    which = np.zeros((2, 2, 2, 3), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"which": which, "cases": cases}))

    # Input 8: 3D array cases, scalar bool which, uint8
    cases = np.full((2, 3, 4), 255, dtype=np.uint8)
    which = np.array(False, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"which": which, "cases": cases}))

    # Input 9: 1D array cases, 1D bool which (same shape), int64
    cases = np.array([-100, 200, -300, 400], dtype=np.int64)
    which = np.zeros((4,), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"which": which, "cases": cases}))

    # Input 10: 2D array cases, scalar int32 which, float16
    cases = np.array([[-1.5, 2.5], [3.5, -4.5]], dtype=np.float16)
    which = np.array(0, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"which": which, "cases": cases}))

    # Input 11: 5D array cases, 5D int32 which (same shape), int8
    cases = np.ones((1, 2, 1, 3, 2), dtype=np.int8) * -7
    which = np.zeros((1, 2, 1, 3, 2), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"which": which, "cases": cases}))

    return list_of_inputs


generated_inputs["jax.lax.select_n"] = select_n_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.select_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.select_n'.")


check_valid('jax.lax.select_n', generated_inputs['jax.lax.select_n'], lib="jax", suffix=0)
