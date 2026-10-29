
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reduce_sum_inputs():
    list_of_inputs = []

    # Input 1: 1D array, reduce along axis 0
    operand = np.array([1.0, -2.0, 3.5, 4.0], dtype=np.float32)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 2: 2D array, reduce along axis 1
    operand = np.arange(12, dtype=np.int32).reshape(3, 4) - 5
    axes = [1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 3: 2D array, reduce along all axes [0, 1]
    operand = np.random.randn(5, 5).astype(np.float64)
    axes = [0, 1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 4: 3D array, reduce along axis 0 and 2
    operand = np.random.uniform(-10, 10, size=(2, 3, 4)).astype(np.float32)
    axes = [0, 2]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 5: 3D array, empty axes list (no reduction)
    operand = np.ones((2, 2, 2), dtype=np.int64)
    axes = []
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 6: 4D array, reduce along a single middle axis
    operand = np.random.randn(2, 4, 3, 5).astype(np.float32)
    axes = [2]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 7: Complex numbers
    operand = (
        np.array([[1.0 + 2.0j, -3.0 + 4.0j], [0.5 - 1.5j, 2.0 + 0.0j]])
    ).astype(np.complex64)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 8: 5D array, reduce along multiple axes
    operand = np.random.randn(2, 2, 3, 2, 4).astype(np.float32)
    axes = [1, 3, 4]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 9: 16-bit float dtype
    operand = np.array([[-1.5, 2.5], [3.0, -4.0]], dtype=np.float16)
    axes = [1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 10: 2D int16 array, reduce along axis 0
    operand = np.array([[-10, 20, 30], [5, -15, 25]], dtype=np.int16)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 11: 4D array, reduce all axes
    operand = np.random.randn(3, 2, 2, 3).astype(np.float64)
    axes = [0, 1, 2, 3]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    return list_of_inputs


generated_inputs["jax.lax.reduce_sum_1"] = reduce_sum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_sum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_sum_1'.")


check_valid('jax.lax.reduce_sum', generated_inputs['jax.lax.reduce_sum_1'], lib="jax", suffix=1)
