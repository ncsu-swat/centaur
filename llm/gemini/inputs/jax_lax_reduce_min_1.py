
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reduce_min_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduce over axis 0
    operand = np.array([-3.5, 1.2, -7.8, 4.0, 0.0], dtype=np.float32)
    axes = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 2: 2D float32 array, reduce over axis 1
    operand = np.random.randn(4, 5).astype(np.float32)
    axes = (1,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 3: 2D int32 array with negative values, reduce over axis 0
    operand = np.array([[-10, 20, -30], [5, -15, 25]], dtype=np.int32)
    axes = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 4: 3D float64 array, reduce over multiple axes (0, 2)
    operand = np.random.randn(3, 4, 5).astype(np.float64)
    axes = (0, 2)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 5: 3D int64 array, reduce over axis 1
    operand = np.random.randint(-100, 100, size=(2, 6, 3), dtype=np.int64)
    axes = (1,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 6: 4D float32 array, reduce over all axes
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axes = (0, 1, 2, 3)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 7: 2D float32 array, reduce over zero axes (empty tuple)
    operand = np.random.randn(3, 3).astype(np.float32)
    axes = ()
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 8: 1D int32 array with large range
    operand = np.array([-32000, 0, 32000, -15000], dtype=np.int32)
    axes = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 9: 5D float32 array, reduce over non-consecutive axes (1, 3)
    operand = np.random.randn(2, 3, 2, 4, 2).astype(np.float32)
    axes = (1, 3)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 10: 3D int32 array, reduce over axes (0, 1)
    operand = np.random.randint(-50, 50, size=(4, 4, 2), dtype=np.int32)
    axes = (0, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 11: 2D float64 array, reduce over axis 0
    operand = np.random.randn(5, 7).astype(np.float64)
    axes = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    return list_of_inputs


generated_inputs["jax.lax.reduce_min_1"] = reduce_min_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_min_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_min_1'.")


check_valid('jax.lax.reduce_min', generated_inputs['jax.lax.reduce_min_1'], lib="jax", suffix=1)
