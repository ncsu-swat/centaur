
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reshape_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D int32
    operand = np.arange(6, dtype=np.int32)
    new_sizes = [2, 3]
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 2: 2D to 1D with permuted dimensions float32
    operand = np.random.randn(2, 3).astype(np.float32)
    new_sizes = [6]
    dimensions = (1, 0)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 3: 2D to 2D same shape identity permutation
    operand = np.random.randn(3, 4).astype(np.float64)
    new_sizes = [3, 4]
    dimensions = (0, 1)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 4: 3D to 2D int64
    operand = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    new_sizes = [6, 4]
    dimensions = (0, 1, 2)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 5: 3D to 3D with reversed dimensions permutation
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    new_sizes = [4, 3, 2]
    dimensions = (2, 1, 0)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 6: 1D to 3D with negative values
    operand = np.linspace(-10.0, 10.0, 24, dtype=np.float32)
    new_sizes = [2, 3, 4]
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 7: 4D to 2D with permuted axes
    operand = np.random.randn(2, 2, 2, 2).astype(np.float32)
    new_sizes = [4, 4]
    dimensions = (0, 2, 1, 3)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 8: 0D scalar to 1D
    operand = np.array(42.0, dtype=np.float32)
    new_sizes = [1]
    dimensions = ()
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 9: 1D size 1 to 0D scalar
    operand = np.array([-5], dtype=np.int32)
    new_sizes = []
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 10: 2D boolean array flattened
    operand = np.array([[True, False], [False, True]], dtype=np.bool_)
    new_sizes = [4]
    dimensions = (0, 1)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    # Input 11: 3D complex64 to 2D
    operand = (
        np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)
    ).astype(np.complex64)
    new_sizes = [8, 3]
    dimensions = (1, 0, 2)
    list_of_inputs.append(
        copy.deepcopy(
            {"operand": operand, "new_sizes": new_sizes, "dimensions": dimensions}
        )
    )

    return list_of_inputs


generated_inputs["jax.lax.reshape_3"] = reshape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reshape_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reshape_3'.")


check_valid('jax.lax.reshape', generated_inputs['jax.lax.reshape_3'], lib="jax", suffix=3)
