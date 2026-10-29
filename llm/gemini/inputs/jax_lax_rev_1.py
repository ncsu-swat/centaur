
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rev_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reversing axis 0
    operand = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    dimensions = [0]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 2: 1D int32 array, empty dimensions (no-op)
    operand = np.array([-10, 0, 10, 20], dtype=np.int32)
    dimensions = []
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 3: 2D float64 array with negative values, reversing axis 0
    operand = np.array(
        [[-1.5, 2.5, -3.5], [4.5, -5.5, 6.5]], dtype=np.float64
    )
    dimensions = [0]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 4: 2D int64 array, reversing axis 1
    operand = np.arange(12, dtype=np.int64).reshape(3, 4)
    dimensions = [1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 5: 2D bool array, reversing both axes [0, 1]
    operand = np.array(
        [[True, False, True], [False, True, False]], dtype=np.bool_
    )
    dimensions = [0, 1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 6: 3D float32 array, reversing axis 2
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    dimensions = [2]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 7: 3D complex64 array, reversing axes 0 and 2
    operand = (
        np.random.randn(3, 2, 4) + 1j * np.random.randn(3, 2, 4)
    ).astype(np.complex64)
    dimensions = [0, 2]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 8: 4D float32 array, reversing axes 1 and 3
    operand = np.random.randn(2, 4, 3, 5).astype(np.float32)
    dimensions = [1, 3]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 9: 4D uint8 array, reversing all axes [0, 1, 2, 3]
    operand = np.arange(24, dtype=np.uint8).reshape(2, 3, 2, 2)
    dimensions = [0, 1, 2, 3]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 10: 0D scalar float32 array, empty dimensions
    operand = np.array(3.14, dtype=np.float32)
    dimensions = []
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 11: 5D int16 array, reversing subset of axes
    operand = np.ones((2, 1, 3, 1, 2), dtype=np.int16)
    dimensions = [0, 2, 4]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    return list_of_inputs


generated_inputs["jax.lax.rev_1"] = rev_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.rev_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.rev_1'.")


check_valid('jax.lax.rev', generated_inputs['jax.lax.rev_1'], lib="jax", suffix=1)
