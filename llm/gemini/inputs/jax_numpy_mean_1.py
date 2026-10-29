
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def mean_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    a = np.array([1.0, -2.0, 3.5, 4.0, -5.5], dtype=np.float32)
    where = np.array([True, True, False, True, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 0,
                "dtype": np.dtype(np.float32),
                "keepdims": False,
                "where": where,
            }
        )
    )

    # Input 2: 2D float64 array, keepdims=True, axis=1
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    where = np.ones((2, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 1,
                "dtype": np.dtype(np.float64),
                "keepdims": True,
                "where": where,
            }
        )
    )

    # Input 3: 2D int32 array, axis=0, reduced to float32
    a = np.array([[-10, 20, -30], [40, -50, 60]], dtype=np.int32)
    where = np.array([[True, False, True], [True, True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 0,
                "dtype": np.dtype(np.float32),
                "keepdims": False,
                "where": where,
            }
        )
    )

    # Input 4: 3D float32 array, negative axis=-1, keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    where = np.random.choice([True, False], size=(2, 3, 4), p=[0.8, 0.2])
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": -1,
                "dtype": np.dtype(np.float32),
                "keepdims": True,
                "where": where,
            }
        )
    )

    # Input 5: 3D complex64 array, axis=1
    a = (np.random.randn(3, 4, 2) + 1j * np.random.randn(3, 4, 2)).astype(
        np.complex64
    )
    where = np.ones((3, 4, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 1,
                "dtype": np.dtype(np.complex64),
                "keepdims": False,
                "where": where,
            }
        )
    )

    # Input 6: 4D float32 array with broadcasting where condition
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    where = np.ones((1, 3, 1, 5), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 2,
                "dtype": np.dtype(np.float64),
                "keepdims": False,
                "where": where,
            }
        )
    )

    # Input 7: 1D int64 array, negative axis=-1, keepdims=True
    a = np.array([-100, 200, -300, 400], dtype=np.int64)
    where = np.array([True, True, True, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": -1,
                "dtype": np.dtype(np.float64),
                "keepdims": True,
                "where": where,
            }
        )
    )

    # Input 8: 2D float16 array, axis=0, keepdims=True
    a = np.array([[-1.5, 2.5], [3.5, -4.5]], dtype=np.float16)
    where = np.array([[True, False], [True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 0,
                "dtype": np.dtype(np.float32),
                "keepdims": True,
                "where": where,
            }
        )
    )

    # Input 9: 3D float64 array, axis=0, keepdims=False
    a = np.random.uniform(-10.0, 10.0, size=(4, 2, 3)).astype(np.float64)
    where = a > 0.0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 0,
                "dtype": np.dtype(np.float64),
                "keepdims": False,
                "where": where,
            }
        )
    )

    # Input 10: 4D float32 array, negative axis=-2, keepdims=True
    a = np.random.randn(2, 2, 3, 4).astype(np.float32)
    where = np.ones((2, 2, 3, 4), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": -2,
                "dtype": np.dtype(np.float32),
                "keepdims": True,
                "where": where,
            }
        )
    )

    # Input 11: 2D boolean array as input, axis=1, keepdims=False
    a = np.array([[True, False, True], [False, False, True]], dtype=bool)
    where = np.ones((2, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": 1,
                "dtype": np.dtype(np.float32),
                "keepdims": False,
                "where": where,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.mean_1"] = mean_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.mean_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.mean_1'.")


check_valid('jax.numpy.mean', generated_inputs['jax.numpy.mean_1'], lib="jax", suffix=1)
