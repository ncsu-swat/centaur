
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expand_dims_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array expanded to 1D
    array = np.array(3.14, dtype=np.float32)
    dimensions = [0]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 2: 0D scalar array expanded to 2D
    array = np.array(42, dtype=np.int32)
    dimensions = [0, 1]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 3: 1D array expanded at front
    array = np.arange(5, dtype=np.float64)
    dimensions = [0]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 4: 1D array expanded at back
    array = np.array([True, False, True], dtype=np.bool_)
    dimensions = [1]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 5: 1D array expanded at both ends
    array = np.array([1, 2, 3, 4], dtype=np.int64)
    dimensions = [0, 2]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 6: 2D array expanded at front
    array = np.random.randn(3, 4).astype(np.float32)
    dimensions = [0]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 7: 2D array expanded in the middle
    array = np.random.randint(0, 10, size=(4, 5), dtype=np.int32)
    dimensions = [1]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 8: 2D array expanded at the end
    array = np.ones((2, 6), dtype=np.uint8)
    dimensions = [2]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 9: 2D array with multiple dimensions inserted
    array = np.random.randn(3, 3).astype(np.float32)
    dimensions = [0, 2, 4]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 10: 3D array expanded at multiple intermediate axes
    array = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(
        np.complex64
    )
    dimensions = [1, 3]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 11: 3D array with empty dimensions list
    array = np.zeros((2, 2, 2), dtype=np.float32)
    dimensions = []
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 12: 4D array expanded to 6D
    array = np.random.randn(2, 1, 3, 4).astype(np.float32)
    dimensions = [0, 5]
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    return list_of_inputs


generated_inputs["jax.lax.expand_dims_1"] = expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.expand_dims_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.expand_dims_1'.")


check_valid('jax.lax.expand_dims', generated_inputs['jax.lax.expand_dims_1'], lib="jax", suffix=1)
