
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reduce_max_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduce along axis 0
    operand = np.array([-1.5, 2.3, 0.0, -4.2, 5.7], dtype=np.float32)
    axes = (0,)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 2: 2D float64 array with negative numbers, reduce along axis 1
    operand = np.random.randn(4, 6).astype(np.float64) - 2.0
    axes = (1,)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 3: 2D int32 array, reduce along axis 0
    operand = np.random.randint(-50, 50, size=(3, 5), dtype=np.int32)
    axes = (0,)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 4: 3D float32 array, reduce over multiple axes (0, 2)
    operand = np.random.randn(2, 4, 3).astype(np.float32)
    axes = (0, 2)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 5: 3D int64 array, reduce over empty axes
    operand = np.random.randint(0, 100, size=(2, 3, 4), dtype=np.int64)
    axes = ()
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 6: 4D float32 array, reduce over axes (1, 2, 3)
    operand = np.random.uniform(-10.0, 10.0, size=(3, 2, 4, 5)).astype(
        np.float32
    )
    axes = (1, 2, 3)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 7: 4D int32 array, reduce over axes (0, 3)
    operand = np.random.randint(-100, 100, size=(2, 5, 3, 2), dtype=np.int32)
    axes = (0, 3)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 8: 2D float64 array, reduce over all axes (0, 1)
    operand = np.array([[1.0, -2.0, 3.5], [-4.0, 5.2, -1.0]], dtype=np.float64)
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 9: 5D float32 array, reduce over axes (2, 4)
    operand = np.random.randn(2, 2, 3, 2, 4).astype(np.float32)
    axes = (2, 4)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 10: 1D int64 array with strictly negative values, reduce along axis 0
    operand = np.array([-100, -20, -5, -45, -1], dtype=np.int64)
    axes = (0,)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    # Input 11: 3D float32 array, reduce over all axes (0, 1, 2)
    operand = np.random.randn(4, 4, 4).astype(np.float32)
    axes = (0, 1, 2)
    list_of_inputs.append(copy.deepcopy({"operand": operand, "axes": axes}))

    return list_of_inputs


generated_inputs["jax.lax.reduce_max_2"] = reduce_max_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_max_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_max_2'.")


check_valid('jax.lax.reduce_max', generated_inputs['jax.lax.reduce_max_2'], lib="jax", suffix=2)
