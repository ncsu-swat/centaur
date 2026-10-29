
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanmedian_inputs():
    list_of_inputs = []

    # Input 1: 1D array with NaN values, axis=0
    a = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 0,
                "keepdims": False,
                "overwrite_input": False,
            }
        )
    )

    # Input 2: 2D array, axis=1, keepdims=True
    a = np.array(
        [[2.0, np.nan, 7.0, np.nan], [np.nan, 5.0, 9.0, 2.0], [6.0, 1.0, np.nan, 3.0]],
        dtype=np.float32,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 1,
                "keepdims": True,
                "overwrite_input": False,
            }
        )
    )

    # Input 3: 2D array, axis=0, keepdims=False, float64
    a = np.array(
        [
            [-10.5, np.nan, 3.2],
            [np.nan, -2.1, 0.0],
            [1.5, -4.3, np.nan],
            [5.0, 6.2, -1.1],
        ],
        dtype=np.float64,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 0,
                "keepdims": False,
                "overwrite_input": False,
            }
        )
    )

    # Input 4: Negative axis (-1) on a 2D array
    a = np.array(
        [[np.nan, np.nan], [1.0, 2.0], [-5.0, np.nan]],
        dtype=np.float32,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": -1,
                "keepdims": True,
                "overwrite_input": False,
            }
        )
    )

    # Input 5: 3D array, axis=1, with negative and positive values
    a = np.array(
        [
            [[-1.0, 2.0], [np.nan, -4.0]],
            [[5.0, np.nan], [7.0, 8.0]],
            [[np.nan, np.nan], [-2.0, 3.0]],
        ],
        dtype=np.float32,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 1,
                "keepdims": False,
                "overwrite_input": False,
            }
        )
    )

    # Input 6: 3D array, axis=2, keepdims=True
    a = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    a[0, 1, 2] = np.nan
    a[1, 2, 0] = np.nan
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 2,
                "keepdims": True,
                "overwrite_input": False,
            }
        )
    )

    # Input 7: 3D array, axis=0, keepdims=False
    a = np.array(
        [
            [[np.nan, 1.0], [2.0, np.nan]],
            [[3.0, 4.0], [np.nan, 6.0]],
            [[np.nan, np.nan], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 0,
                "keepdims": False,
                "overwrite_input": False,
            }
        )
    )

    # Input 8: 4D array, axis=3, keepdims=False
    a = np.arange(16, dtype=np.float32).reshape((2, 2, 2, 2))
    a[a % 3 == 0] = np.nan
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 3,
                "keepdims": False,
                "overwrite_input": False,
            }
        )
    )

    # Input 9: 4D array, negative axis=-2, keepdims=True
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    a[a < 0] = np.nan
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": -2,
                "keepdims": True,
                "overwrite_input": False,
            }
        )
    )

    # Input 10: 1D array with all NaNs, axis=-1, keepdims=False
    a = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": -1,
                "keepdims": False,
                "overwrite_input": False,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanmedian_1"] = nanmedian_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanmedian_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanmedian_1'.")


check_valid('jax.numpy.nanmedian', generated_inputs['jax.numpy.nanmedian_1'], lib="jax", suffix=1)
