
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nanquantile_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with NaN, median q=0.5
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": 0.5,
                "axis": 0,
                "overwrite_input": False,
                "method": "inverted_cdf",
                "keepdims": False,
                "weights": weights,
            }
        )
    )

    # Input 2: 2D float32 array, axis=1, keepdims=True
    a = np.array(
        [[np.nan, 2.0, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    weights = np.ones((3, 3), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": 0.25,
                "axis": 1,
                "overwrite_input": False,
                "method": "inverted_cdf",
                "keepdims": True,
                "weights": weights,
            }
        )
    )

    # Input 3: 2D float64 array, axis=0, keepdims=False, q=0.75
    a = np.array(
        [[-10.0, np.nan], [0.0, 5.0], [np.nan, 20.0]], dtype=np.float64
    )
    weights = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": 0.75,
                "axis": 0,
                "overwrite_input": False,
                "method": "inverted_cdf",
                "keepdims": False,
                "weights": weights,
            }
        )
    )

    # Input 4: 3D float32 array, axis=-1, keepdims=True, minimum q=0.0
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    a[0, 1, 2] = np.nan
    a[1, 0, 0] = np.nan
    weights = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": 0.0,
                "axis": -1,
                "overwrite_input": False,
                "method": "inverted_cdf",
                "keepdims": True,
                "weights": weights,
            }
        )
    )

    # Input 5: 3D float64 array, axis=1, keepdims=False, maximum q=1.0
    a = np.array(
        [
            [[1.0, -2.0], [np.nan, 4.0]],
            [[5.0, np.nan], [7.0, -8.0]],
            [[np.nan, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float64,
    )
    weights = np.full((3, 2, 2), 0.5, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": 1.0,
                "axis": 1,
                "overwrite_input": False,
                "method": "inverted_cdf",
                "keepdims": False,
                "weights": weights,
            }
        )
    )

    # Input 6: 1D float64 array with negative values and NaNs, q=0.1
    a = np.array([-100.0, -50.0, np.nan, -10.0, 0.0, 50.0], dtype=np.float64)
    weights = np.array([2.0, 1.0, 1.0, 3.0, 1.0, 2.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": 0.1,
                "axis": 0,
                "overwrite_input": False,
                "method": "inverted_cdf",
                "keepdims": True,
                "weights": weights,
            }
        )
    )

    # Input 7: 2D float32 array, negative axis=-2, q=0.9
    a = np.array(
        [[1.5, 2.5, np.nan], [np.nan, 5.5, 6.5], [7.5, np.nan, 9.5]],
        dtype=np.float32,
    )
    weights = np.array([[1.0, 2.0, 1.0]], dtype=np.float32)
    weights = np.broadcast_to(weights, a.shape).copy()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": 0.9,
                "axis": -2,
                "overwrite_input": False,
                "method": "inverted_cdf",
                "keepdims": False,
                "weights": weights,
            }
        )
    )

    # Input 8: 4D float32 array, axis=2, keepdims=True, q=0.4
    a = np.ones((2, 2, 4, 3), dtype=np.float32)
    a[0, 0, 1, 1] = np.nan
    a[1, 1, 2, 0] = np.nan
    weights = np.full((2, 2, 4, 3), 1.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": 0.4,
                "axis": 2,
                "overwrite_input": False,
                "method": "inverted_cdf",
                "keepdims": True,
                "weights": weights,
            }
        )
    )

    # Input 9: 2D float64 array with varying weights, axis=0, keepdims=True, q=0.6
    a = np.array(
        [[np.nan, 12.0], [-3.0, np.nan], [4.5, 8.2], [1.1, -2.3]],
        dtype=np.float64,
    )
    weights = np.array(
        [[1.0, 2.0], [3.0, 1.0], [2.0, 2.0], [1.0, 4.0]], dtype=np.float64
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": 0.6,
                "axis": 0,
                "overwrite_input": False,
                "method": "inverted_cdf",
                "keepdims": True,
                "weights": weights,
            }
        )
    )

    # Input 10: 3D float32 array, axis=0, keepdims=False, q=0.33
    a = np.array(
        [
            [[10.0, 20.0], [np.nan, 40.0]],
            [[50.0, np.nan], [70.0, 80.0]],
            [[np.nan, 100.0], [110.0, 120.0]],
            [[130.0, 140.0], [150.0, np.nan]],
        ],
        dtype=np.float32,
    )
    weights = np.ones((4, 2, 2), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "q": 0.33,
                "axis": 0,
                "overwrite_input": False,
                "method": "inverted_cdf",
                "keepdims": False,
                "weights": weights,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.nanquantile_3"] = nanquantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nanquantile_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nanquantile_3'.")


check_valid('jax.numpy.nanquantile', generated_inputs['jax.numpy.nanquantile_3'], lib="jax", suffix=3)
