
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D reshape
    operand = np.arange(6, dtype=np.int32)
    new_sizes = (2, 3)
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 2: 2D to 1D with permuted dimensions
    operand = np.arange(6, dtype=np.float32).reshape(2, 3)
    new_sizes = (6,)
    dimensions = (1, 0)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 3: 2D to 2D with standard ordering and negative values
    operand = np.array([[-1.5, 2.0, -3.2, 4.1], [5.0, -6.3, 7.2, -8.0]], dtype=np.float64)
    new_sizes = (4, 2)
    dimensions = (0, 1)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 4: 3D to 2D with non-trivial permutation
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    new_sizes = (6, 4)
    dimensions = (1, 0, 2)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 5: 0D scalar to 1D array
    operand = np.array(42, dtype=np.int64)
    new_sizes = (1,)
    dimensions = ()
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 6: 1D array of size 1 to 0D scalar
    operand = np.array([-3.14], dtype=np.float32)
    new_sizes = ()
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 7: 4D to 1D with complete dimension reversal
    operand = np.arange(16, dtype=np.int32).reshape(2, 2, 2, 2)
    new_sizes = (16,)
    dimensions = (3, 2, 1, 0)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 8: 3D to 3D with permutation
    operand = np.random.randn(2, 4, 3).astype(np.float32)
    new_sizes = (4, 3, 2)
    dimensions = (1, 2, 0)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 9: Boolean array reshape
    operand = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    new_sizes = (3, 2)
    dimensions = (1, 0)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 10: Complex numbers reshape
    operand = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex64)
    new_sizes = (6,)
    dimensions = (0, 1)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 11: Reshaping empty array with 0-size dimension
    operand = np.empty((0, 4), dtype=np.float32)
    new_sizes = (0, 2, 2)
    dimensions = (0, 1)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    return list_of_inputs


generated_inputs["jax.lax.reshape_1"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reshape_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reshape_1'.")


check_valid('jax.lax.reshape', generated_inputs['jax.lax.reshape_1'], lib="jax", suffix=1)
