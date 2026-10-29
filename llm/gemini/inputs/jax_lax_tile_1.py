
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tile_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    operand = np.array([1, -2, 3], dtype=np.int32)
    reps = (3,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": reps})
    )

    # Input 2: 2D float32 array
    operand = np.array([[-1.5, 2.0], [3.2, -4.8]], dtype=np.float32)
    reps = (2, 3)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": reps})
    )

    # Input 3: 2D float64 array with reps (1, 1)
    operand = np.random.randn(4, 5).astype(np.float64)
    reps = (1, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": reps})
    )

    # Input 4: 1D boolean array
    operand = np.array([True, False, True], dtype=np.bool_)
    reps = (4,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": reps})
    )

    # Input 5: 3D float32 array
    operand = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 12.0
    reps = (2, 1, 3)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": reps})
    )

    # Input 6: 4D int16 array
    operand = np.ones((2, 1, 3, 2), dtype=np.int16) * -5
    reps = (1, 3, 1, 2)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": reps})
    )

    # Input 7: 2D complex64 array
    operand = np.array([[1 + 2j, -3 - 4j], [5j, -6]], dtype=np.complex64)
    reps = (3, 2)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": reps})
    )

    # Input 8: 0D scalar float32 array
    operand = np.array(3.14, dtype=np.float32)
    reps = ()
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": reps})
    )

    # Input 9: 5D uint8 array
    operand = np.full((1, 2, 1, 2, 1), 255, dtype=np.uint8)
    reps = (2, 1, 2, 1, 3)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": reps})
    )

    # Input 10: 3D int64 array with zero/negative integers
    operand = np.array([[[-10, 0], [20, -30]]], dtype=np.int64)
    reps = (3, 2, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": reps})
    )

    # Input 11: 1D float32 array with zeros
    operand = np.zeros((10,), dtype=np.float32)
    reps = (5,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "reps": reps})
    )

    return list_of_inputs


generated_inputs["jax.lax.tile_1"] = tile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tile_1'.")


check_valid('jax.lax.tile', generated_inputs['jax.lax.tile_1'], lib="jax", suffix=1)
