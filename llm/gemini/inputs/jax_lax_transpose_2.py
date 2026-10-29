
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 standard matrix transpose
    operand = np.random.randn(3, 5).astype(np.float32)
    permutation = (1, 0)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 2: 2D int32 identity transpose
    operand = np.random.randint(-10, 10, size=(4, 4), dtype=np.int32)
    permutation = (0, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 3: 3D float64 reorder axes
    operand = np.random.randn(2, 3, 4).astype(np.float64)
    permutation = (2, 0, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 4: 1D array transpose
    operand = np.array([-1.5, 0.0, 2.3, -4.1], dtype=np.float32)
    permutation = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 5: 0D scalar array
    operand = np.array(42, dtype=np.int64)
    permutation = ()
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 6: 4D uint8 reverse all axes
    operand = np.random.randint(0, 256, size=(2, 3, 4, 5), dtype=np.uint8)
    permutation = (3, 2, 1, 0)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 7: 3D boolean array
    operand = np.array(
        [[[True, False], [False, True]], [[False, False], [True, True]]],
        dtype=bool,
    )
    permutation = (1, 2, 0)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 8: 2D complex64 array with negative components
    operand = (
        np.random.randn(3, 2) + 1j * np.random.randn(3, 2)
    ).astype(np.complex64)
    permutation = (1, 0)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 9: 5D float16 array
    operand = np.random.randn(2, 1, 3, 1, 4).astype(np.float16)
    permutation = (4, 2, 0, 3, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    # Input 10: 3D int16 array with negative values
    operand = np.random.randint(-100, 100, size=(5, 2, 7), dtype=np.int16)
    permutation = (0, 2, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "permutation": permutation})
    )

    return list_of_inputs


generated_inputs["jax.lax.transpose_2"] = transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.transpose_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.transpose_2'.")


check_valid('jax.lax.transpose', generated_inputs['jax.lax.transpose_2'], lib="jax", suffix=2)
