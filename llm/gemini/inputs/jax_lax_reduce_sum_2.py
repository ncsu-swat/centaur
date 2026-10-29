
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reduce_sum_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduce over axis 0
    operand = np.array([1.0, -2.5, 3.5, 4.0], dtype=np.float32)
    axes = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 2: 2D float32 array, reduce over axis 0
    operand = np.random.randn(3, 4).astype(np.float32)
    axes = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 3: 2D float32 array, reduce over axis 1
    operand = np.random.randn(5, 6).astype(np.float32)
    axes = (1,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 4: 2D int32 array with negative values, reduce over all axes
    operand = np.array([[-10, 20, -30], [40, -50, 60]], dtype=np.int32)
    axes = (0, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 5: 3D float64 array, reduce over middle axis
    operand = np.random.randn(2, 3, 4).astype(np.float64)
    axes = (1,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 6: 3D float32 array, reduce over non-adjacent axes
    operand = np.random.randn(4, 5, 6).astype(np.float32)
    axes = (0, 2)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 7: 4D float32 array, empty axes tuple (no reduction)
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axes = ()
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 8: 4D int64 array, reduce over last two axes
    operand = np.arange(24, dtype=np.int64).reshape((2, 3, 2, 2)) - 10
    axes = (2, 3)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 9: 2D complex64 array, reduce over axis 0
    operand = (
        np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    ).astype(np.complex64)
    axes = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 10: 5D float16 array, reduce over multiple axes
    operand = np.random.randn(2, 2, 3, 2, 4).astype(np.float16)
    axes = (1, 3, 4)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 11: 0D scalar array, empty axes reduction
    operand = np.array(42.0, dtype=np.float32)
    axes = ()
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    return list_of_inputs


generated_inputs["jax.lax.reduce_sum_2"] = reduce_sum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_sum_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_sum_2'.")


check_valid('jax.lax.reduce_sum', generated_inputs['jax.lax.reduce_sum_2'], lib="jax", suffix=2)
