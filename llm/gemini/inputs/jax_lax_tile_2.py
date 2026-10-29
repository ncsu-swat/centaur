
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tile_inputs():
    list_of_inputs = []

    # Input 1: 1D array, int32
    operand = np.array([1, 2, 3], dtype=np.int32)
    reps = [3]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": list(reps)})
    )

    # Input 2: 2D array, float32, negative values
    operand = np.array([[-1.5, 2.0], [0.5, -3.2]], dtype=np.float32)
    reps = [2, 3]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": list(reps)})
    )

    # Input 3: 0D array (scalar tensor), int64
    operand = np.array(42, dtype=np.int64)
    reps = []
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": list(reps)})
    )

    # Input 4: 3D array, float64
    operand = np.random.randn(2, 3, 4).astype(np.float64)
    reps = [1, 2, 1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": list(reps)})
    )

    # Input 5: 1D boolean array
    operand = np.array([True, False, True], dtype=np.bool_)
    reps = [4]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": list(reps)})
    )

    # Input 6: 4D array, complex64
    operand = (
        np.random.randn(1, 2, 1, 3) + 1j * np.random.randn(1, 2, 1, 3)
    ).astype(np.complex64)
    reps = [2, 1, 3, 1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": list(reps)})
    )

    # Input 7: 2D array with 1 repetition along all dimensions
    operand = np.arange(12, dtype=np.int16).reshape(3, 4)
    reps = [1, 1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": list(reps)})
    )

    # Input 8: 2D array with reps containing 0
    operand = np.ones((2, 2), dtype=np.float32)
    reps = [0, 2]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": list(reps)})
    )

    # Input 9: 5D array, uint8
    operand = np.zeros((1, 1, 2, 1, 2), dtype=np.uint8)
    reps = [2, 3, 1, 2, 1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": list(reps)})
    )

    # Input 10: 2D column vector tiled horizontally and vertically
    operand = np.array([[-10], [20], [-30]], dtype=np.int32)
    reps = [2, 5]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": list(reps)})
    )

    # Input 11: 3D array with zeros in reps
    operand = np.random.randn(3, 1, 2).astype(np.float32)
    reps = [2, 0, 3]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": list(reps)})
    )

    return list_of_inputs


generated_inputs["jax.lax.tile_2"] = tile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tile_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tile_2'.")


check_valid('jax.lax.tile', generated_inputs['jax.lax.tile_2'], lib="jax", suffix=2)
