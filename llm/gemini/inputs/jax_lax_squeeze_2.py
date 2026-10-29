
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def squeeze_inputs():
    list_of_inputs = []

    # Input 1: 1D array squeezed to 0D
    array = np.array([3.14], dtype=np.float32)
    dimensions = [0]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 2: 2D array, squeeze first dimension
    array = np.random.randn(1, 5).astype(np.float32)
    dimensions = [0]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 3: 2D array, squeeze second dimension
    array = np.random.randint(0, 10, size=(4, 1), dtype=np.int32)
    dimensions = [1]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 4: 3D array, squeeze multiple dimensions
    array = np.random.randn(1, 1, 1).astype(np.float64)
    dimensions = [0, 1, 2]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 5: 4D array, squeeze alternating size-1 dimensions
    array = np.random.randn(2, 1, 3, 1).astype(np.float32)
    dimensions = [1, 3]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 6: 5D array, squeeze subset of size-1 dimensions
    array = np.zeros((1, 4, 1, 5, 1), dtype=np.int64)
    dimensions = [0, 2]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 7: Empty dimensions list (no-op squeeze)
    array = np.random.randn(2, 3).astype(np.float32)
    dimensions = []
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 8: 2D boolean array, squeeze single dimension
    array = np.array([[True], [False]], dtype=np.bool_)
    dimensions = [1]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 9: 3D complex array, squeeze middle dimension
    array = (np.random.randn(5, 1, 2) + 1j * np.random.randn(5, 1, 2)).astype(
        np.complex64
    )
    dimensions = [1]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 10: 4D uint8 array, squeeze non-consecutive dimensions
    array = np.ones((1, 1, 1, 1), dtype=np.uint8)
    dimensions = [0, 2]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 11: 3D int16 array, squeeze trailing dimensions
    array = np.arange(6, dtype=np.int16).reshape(6, 1, 1)
    dimensions = [1, 2]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    return list_of_inputs


generated_inputs["jax.lax.squeeze_2"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.squeeze_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.squeeze_2'.")


check_valid('jax.lax.squeeze', generated_inputs['jax.lax.squeeze_2'], lib="jax", suffix=2)
