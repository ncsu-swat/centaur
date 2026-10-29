
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def lax_reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D float32
    operand = np.arange(6, dtype=np.float32)
    new_sizes = (2, 3)
    dimensions = [0]
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 2: 2D to 1D int32 with dimension permutation
    operand = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    new_sizes = (6,)
    dimensions = [1, 0]
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 3: 2D to 2D float64
    operand = np.linspace(-5.0, 5.0, 12, dtype=np.float64).reshape(3, 4)
    new_sizes = (2, 6)
    dimensions = [0, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 4: 3D to 2D int64 with axis permutation
    operand = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    new_sizes = (4, 6)
    dimensions = [2, 0, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 5: 4D to 2D float32 reversing all axes
    operand = np.random.randn(2, 2, 2, 2).astype(np.float32)
    new_sizes = (4, 4)
    dimensions = [3, 2, 1, 0]
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 6: 1D boolean array to 2D
    operand = np.array([True, False, False, True, True, False], dtype=np.bool_)
    new_sizes = (3, 2)
    dimensions = [0]
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 7: 3D to 1D complex64 with permutation
    operand = (np.arange(6) + 1j * np.arange(6)).astype(np.complex64).reshape(2, 1, 3)
    new_sizes = (6,)
    dimensions = [1, 2, 0]
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 8: 3D to 1D with negative float32 values
    operand = np.array([[[-1.5, -2.5], [3.5, 4.5], [-5.5, 6.5]],
                        [[7.5, -8.5], [-9.5, 10.5], [11.5, -12.5]]], dtype=np.float32)
    new_sizes = (12,)
    dimensions = [0, 2, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 9: 2D to 3D int16 with transposed input
    operand = np.arange(20, dtype=np.int16).reshape(4, 5)
    new_sizes = (2, 2, 5)
    dimensions = [1, 0]
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 10: 2D uint8 to 3D
    operand = np.arange(8, dtype=np.uint8).reshape(1, 8)
    new_sizes = (2, 2, 2)
    dimensions = [0, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 11: 4D to 2D float64 with mixed permutation
    operand = np.random.randn(2, 3, 2, 2).astype(np.float64)
    new_sizes = (8, 3)
    dimensions = [3, 0, 2, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    return list_of_inputs

generated_inputs["jax.lax.reshape_2"] = lax_reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reshape_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reshape_2'.")


check_valid('jax.lax.reshape', generated_inputs['jax.lax.reshape_2'], lib="jax", suffix=2)
