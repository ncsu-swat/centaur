
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expand_dims_inputs():
    list_of_inputs = []

    # Input 1: 0-D array expanded to 1-D
    array = np.array(3.14, dtype=np.float32)
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 2: 1-D array expanded at the beginning
    array = np.arange(5, dtype=np.int32)
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 3: 1-D array expanded at the end
    array = np.arange(10, dtype=np.float64)
    dimensions = (1,)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 4: 1-D array expanded at multiple positions
    array = np.array([True, False, True], dtype=np.bool_)
    dimensions = (0, 2)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 5: 2-D array expanded in the middle
    array = np.ones((3, 4), dtype=np.float32)
    dimensions = (1,)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 6: 2-D array expanded at both ends
    array = np.zeros((2, 5), dtype=np.int64)
    dimensions = (0, 3)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 7: 3-D array with complex numbers expanded at multiple dimensions
    array = np.random.randn(2, 3, 4).astype(np.complex64)
    dimensions = (1, 3)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 8: 3-D array expanded with empty dimensions tuple
    array = np.ones((2, 2, 2), dtype=np.uint8)
    dimensions = ()
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 9: 4-D array expanded at multiple positions
    array = np.random.randn(2, 3, 2, 4).astype(np.float32)
    dimensions = (0, 2, 5)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 10: 2-D array expanded at consecutive dimensions
    array = np.arange(12, dtype=np.int16).reshape((3, 4))
    dimensions = (1, 2)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    # Input 11: 0-D array expanded to 3-D
    array = np.array(42, dtype=np.int32)
    dimensions = (0, 1, 2)
    list_of_inputs.append(
        copy.deepcopy({"array": array, "dimensions": dimensions})
    )

    return list_of_inputs


generated_inputs["jax.lax.expand_dims_2"] = expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.expand_dims_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.expand_dims_2'.")


check_valid('jax.lax.expand_dims', generated_inputs['jax.lax.expand_dims_2'], lib="jax", suffix=2)
