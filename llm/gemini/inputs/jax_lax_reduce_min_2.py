
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reduce_min_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduce along axis 0
    operand = np.array([-1.5, 2.3, 0.0, -4.7, 3.2], dtype=np.float32)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 2: 2D float32 array, reduce along axis 0
    operand = np.random.randn(3, 4).astype(np.float32)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 3: 2D float64 array, reduce along axis 1
    operand = np.random.randn(5, 6).astype(np.float64)
    axes = [1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 4: 2D int32 array, reduce along all axes
    operand = np.array([[-10, 20, 5], [0, -30, 15]], dtype=np.int32)
    axes = [0, 1]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 5: 3D float32 array, reduce along axes 1 and 2
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    axes = [1, 2]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 6: 3D int64 array with negative numbers, reduce along axis 0
    operand = np.random.randint(-100, 100, size=(4, 2, 3)).astype(np.int64)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 7: 4D float32 array, reduce along non-contiguous axes [0, 2]
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axes = [0, 2]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 8: 1D int32 array, reduce along axis 0
    operand = np.array([10, -5, 3, 0, -20], dtype=np.int32)
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 9: 3D int32 array, reduce along axis 2
    operand = np.random.randint(0, 1000, size=(2, 4, 3)).astype(np.int32)
    axes = [2]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 10: 2D float32 array containing inf and -inf, reduce along axis 0
    operand = np.array(
        [[np.inf, 2.0, -np.inf], [3.0, 1.0, 0.0]], dtype=np.float32
    )
    axes = [0]
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 11: 0D scalar float32 array, empty axes reduction
    operand = np.array(-5.5, dtype=np.float32)
    axes = []
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 12: 3D float64 array, empty axes list (identity operation)
    operand = np.random.randn(2, 2, 2).astype(np.float64)
    axes = []
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    return list_of_inputs


generated_inputs["jax.lax.reduce_min_2"] = reduce_min_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_min_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_min_2'.")


check_valid('jax.lax.reduce_min', generated_inputs['jax.lax.reduce_min_2'], lib="jax", suffix=2)
