
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def squeeze_inputs():
    list_of_inputs = []

    # Input 1: 1D array of shape (1,) squeezed completely to 0D
    array = np.array([3.14], dtype=np.float32)
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 2: 2D array of shape (1, 5), squeeze leading dimension
    array = np.arange(5, dtype=np.int32).reshape(1, 5)
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 3: 2D array of shape (5, 1), squeeze trailing dimension
    array = np.arange(5, dtype=np.int64).reshape(5, 1)
    dimensions = (1,)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 4: 3D array of shape (1, 4, 1), squeeze both size-1 dimensions
    array = np.ones((1, 4, 1), dtype=np.float64)
    dimensions = (0, 2)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 5: 3D array of shape (1, 1, 1), squeeze all dimensions to scalar
    array = np.array([[[True]]], dtype=np.bool_)
    dimensions = (0, 1, 2)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 6: 4D array of shape (2, 1, 3, 1), squeeze only one size-1 dimension
    array = np.zeros((2, 1, 3, 1), dtype=np.uint8)
    dimensions = (1,)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 7: 4D array of shape (2, 1, 3, 1), squeeze both size-1 dimensions
    array = np.random.randn(2, 1, 3, 1).astype(np.float32)
    dimensions = (1, 3)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 8: 2D array, squeeze empty dimensions tuple (no-op)
    array = np.random.randn(3, 4).astype(np.float32)
    dimensions = ()
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 9: 5D array of shape (1, 2, 1, 2, 1), squeeze alternating dimensions
    array = (
        np.arange(4, dtype=np.complex64).reshape(1, 2, 1, 2, 1) + 1j
    ).astype(np.complex64)
    dimensions = (0, 2, 4)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 10: 4D array of shape (1, 1, 7, 1), squeeze subset of size-1 dimensions
    array = np.ones((1, 1, 7, 1), dtype=np.int16)
    dimensions = (0, 3)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 11: 3D array of shape (3, 1, 1), squeeze trailing two dimensions
    array = np.array([[[1.5]], [[2.5]], [[3.5]]], dtype=np.float32)
    dimensions = (1, 2)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    return list_of_inputs


generated_inputs["jax.lax.squeeze_1"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.squeeze_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.squeeze_1'.")


check_valid('jax.lax.squeeze', generated_inputs['jax.lax.squeeze_1'], lib="jax", suffix=1)
