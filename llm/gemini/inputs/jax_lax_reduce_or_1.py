
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reduce_or_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, reduce over axis 0
    operand = np.array([True, False, True, False], dtype=np.bool_)
    axes = [0]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 2: 2D int32 array, reduce over axis 1
    operand = np.array([[1, 2, 4], [8, 16, 32]], dtype=np.int32)
    axes = [1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 3: 2D signed int32 array with negative numbers, reduce over axis 0
    operand = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    axes = [0]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 4: 3D int32 array, reduce over multiple axes (0, 2)
    operand = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    axes = [0, 2]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 5: 3D int64 array, reduce over all axes
    operand = np.array(
        [[[-10, 20], [30, -40]], [[50, -60], [70, 80]]], dtype=np.int64
    )
    axes = [0, 1, 2]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 6: 4D boolean array, reduce over axis 2
    operand = np.random.choice([True, False], size=(2, 3, 4, 5)).astype(np.bool_)
    axes = [2]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 7: 2D int64 array, reduce over empty axes list (no reduction)
    operand = np.array([[10, 20], [30, 40]], dtype=np.int64)
    axes = []
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 8: 1D int32 array, reduce over axis 0
    operand = np.array([0, 15, 240, 3840], dtype=np.int32)
    axes = [0]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 9: 4D int32 array, reduce over axes (1, 3)
    operand = np.random.randint(-100, 100, size=(2, 3, 2, 4), dtype=np.int32)
    axes = [1, 3]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 10: 5D int64 array, reduce over axes (0, 2, 4)
    operand = np.ones((2, 2, 2, 2, 2), dtype=np.int64)
    axes = [0, 2, 4]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 11: 3D boolean array, reduce over axes (0, 1)
    operand = np.array(
        [[[False, True], [False, False]], [[True, False], [False, True]]],
        dtype=np.bool_,
    )
    axes = [0, 1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    return list_of_inputs


generated_inputs["jax.lax.reduce_or_1"] = reduce_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_or_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_or_1'.")


check_valid('jax.lax.reduce_or', generated_inputs['jax.lax.reduce_or_1'], lib="jax", suffix=1)
