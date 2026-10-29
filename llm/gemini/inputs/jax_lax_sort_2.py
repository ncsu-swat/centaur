
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sort_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, sort last dimension
    list_of_inputs.append(
        {
            "operand": np.array([3.0, -1.0, 2.5, 0.0, -4.2], dtype=np.float32),
            "dimension": -1,
            "is_stable": True,
            "num_keys": 1,
        }
    )

    # Input 2: 2D array, int32, sort along axis 0, unstable
    list_of_inputs.append(
        {
            "operand": np.array(
                [[5, 2, -1], [3, 8, 4], [-2, 0, 7], [1, -5, 6]], dtype=np.int32
            ),
            "dimension": 0,
            "is_stable": False,
            "num_keys": 1,
        }
    )

    # Input 3: 2D array, float64, sort along axis 1
    list_of_inputs.append(
        {
            "operand": np.random.randn(4, 6).astype(np.float64),
            "dimension": 1,
            "is_stable": True,
            "num_keys": 1,
        }
    )

    # Input 4: 1D array, int64, negative numbers
    list_of_inputs.append(
        {
            "operand": np.array(
                [-100, 50, -25, 0, 999, -500], dtype=np.int64
            ),
            "dimension": 0,
            "is_stable": True,
            "num_keys": 1,
        }
    )

    # Input 5: 3D array, int32, dimension 2
    list_of_inputs.append(
        {
            "operand": np.random.randint(
                0, 100, size=(2, 3, 5), dtype=np.int32
            ),
            "dimension": 2,
            "is_stable": False,
            "num_keys": 1,
        }
    )

    # Input 6: 3D array, float32, dimension -1
    list_of_inputs.append(
        {
            "operand": np.random.randn(3, 4, 4).astype(np.float32),
            "dimension": -1,
            "is_stable": True,
            "num_keys": 1,
        }
    )

    # Input 7: 1D array, float64, positive and negative floats
    list_of_inputs.append(
        {
            "operand": np.array(
                [1.5, -2.3, 0.0, 4.7, -9.1, 3.2],
                dtype=np.float64,
            ),
            "dimension": -1,
            "is_stable": True,
            "num_keys": 1,
        }
    )

    # Input 8: 4D array, float32, negative dimension -2
    list_of_inputs.append(
        {
            "operand": np.random.randn(2, 3, 4, 5).astype(np.float32),
            "dimension": -2,
            "is_stable": False,
            "num_keys": 1,
        }
    )

    # Input 9: 2D array, int64 values, axis -1
    list_of_inputs.append(
        {
            "operand": np.array([[1, 0, 1], [0, 0, 1]], dtype=np.int64),
            "dimension": -1,
            "is_stable": True,
            "num_keys": 1,
        }
    )

    # Input 10: 2D array, float32 with NaNs, NaNs should sort to the end
    list_of_inputs.append(
        {
            "operand": np.array(
                [[np.nan, 1.0, -2.0], [3.0, np.nan, 0.0]], dtype=np.float32
            ),
            "dimension": 1,
            "is_stable": True,
            "num_keys": 1,
        }
    )

    # Input 11: 3D array, int32, sort along axis 1
    list_of_inputs.append(
        {
            "operand": np.random.randint(
                -50, 50, size=(4, 5, 3), dtype=np.int32
            ),
            "dimension": 1,
            "is_stable": False,
            "num_keys": 1,
        }
    )

    return list_of_inputs


generated_inputs["jax.lax.sort_2"] = sort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sort_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sort_2'.")


check_valid('jax.lax.sort', generated_inputs['jax.lax.sort_2'], lib="jax", suffix=2)
