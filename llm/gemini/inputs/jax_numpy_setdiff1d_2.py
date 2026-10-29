
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def setdiff1d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 arrays with partial overlap
    ar1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ar2 = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "ar1": ar1,
                "ar2": ar2,
                "assume_unique": False,
                "size": 4,
                "fill_value": 0.0,
            }
        )
    )

    # Input 2: Arrays with negative values and size smaller than result (truncation)
    ar1 = np.array([-5.0, -3.0, -1.0, 1.0, 3.0], dtype=np.float32)
    ar2 = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "ar1": ar1,
                "ar2": ar2,
                "assume_unique": False,
                "size": 2,
                "fill_value": -999.0,
            }
        )
    )

    # Input 3: Float64 arrays, size larger than result (padding needed)
    ar1 = np.array([10.5, 20.5, 30.5], dtype=np.float64)
    ar2 = np.array([20.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "ar1": ar1,
                "ar2": ar2,
                "assume_unique": False,
                "size": 5,
                "fill_value": -1.0,
            }
        )
    )

    # Input 4: Completely disjoint arrays, assume_unique=True
    ar1 = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    ar2 = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "ar1": ar1,
                "ar2": ar2,
                "assume_unique": True,
                "size": 4,
                "fill_value": 0.0,
            }
        )
    )

    # Input 5: Identical arrays resulting in empty set difference, padded to size
    ar1 = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    ar2 = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "ar1": ar1,
                "ar2": ar2,
                "assume_unique": True,
                "size": 3,
                "fill_value": 99.0,
            }
        )
    )

    # Input 6: Arrays with duplicate elements, assume_unique=False
    ar1 = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32)
    ar2 = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "ar1": ar1,
                "ar2": ar2,
                "assume_unique": False,
                "size": 3,
                "fill_value": -0.5,
            }
        )
    )

    # Input 7: Empty ar1, non-empty ar2, size=0
    ar1 = np.array([], dtype=np.float32)
    ar2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "ar1": ar1,
                "ar2": ar2,
                "assume_unique": False,
                "size": 0,
                "fill_value": 0.0,
            }
        )
    )

    # Input 8: Non-empty ar1, empty ar2, assume_unique=True
    ar1 = np.array([5.0, 15.0, 25.0], dtype=np.float64)
    ar2 = np.array([], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "ar1": ar1,
                "ar2": ar2,
                "assume_unique": True,
                "size": 3,
                "fill_value": 0.0,
            }
        )
    )

    # Input 9: Single-element arrays, disjoint
    ar1 = np.array([42.0], dtype=np.float32)
    ar2 = np.array([24.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "ar1": ar1,
                "ar2": ar2,
                "assume_unique": True,
                "size": 1,
                "fill_value": 0.0,
            }
        )
    )

    # Input 10: Both arrays empty with size > 0 requiring fill
    ar1 = np.array([], dtype=np.float32)
    ar2 = np.array([], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "ar1": ar1,
                "ar2": ar2,
                "assume_unique": False,
                "size": 3,
                "fill_value": -1.0,
            }
        )
    )

    # Input 11: Larger arrays with mixed ranges
    ar1 = np.linspace(-10.0, 10.0, 21, dtype=np.float32)
    ar2 = np.linspace(-5.0, 5.0, 11, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "ar1": ar1,
                "ar2": ar2,
                "assume_unique": True,
                "size": 10,
                "fill_value": 100.0,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.setdiff1d_2"] = setdiff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.setdiff1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.setdiff1d_2'.")


check_valid('jax.numpy.setdiff1d', generated_inputs['jax.numpy.setdiff1d_2'], lib="jax", suffix=2)
