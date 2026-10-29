
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def stack_inputs():
    list_of_inputs = []

    # Case 1: 1D array, axis=0, float32
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arr, "axis": 0, "dtype": np.dtype(np.float32)})
    )

    # Case 2: 1D array, axis=-1, int32
    arr = np.array([10, 20, 30], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arr, "axis": -1, "dtype": np.dtype(np.int32)})
    )

    # Case 3: 2D array, axis=0, float32
    arr = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arr, "axis": 0, "dtype": np.dtype(np.float32)})
    )

    # Case 4: 2D array, axis=1, int64
    arr = np.arange(10, dtype=np.int64).reshape((2, 5))
    list_of_inputs.append(
        copy.deepcopy({"arrays": arr, "axis": 1, "dtype": np.dtype(np.int64)})
    )

    # Case 5: 2D array, negative axis=-2, float64
    arr = np.random.randn(4, 3).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"arrays": arr, "axis": -2, "dtype": np.dtype(np.float64)}
        )
    )

    # Case 6: 2D array, axis=-1, bool_
    arr = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arr, "axis": -1, "dtype": np.dtype(np.bool_)})
    )

    # Case 7: 3D array, axis=0, complex64
    arr = (
        np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
    ).astype(np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {"arrays": arr, "axis": 0, "dtype": np.dtype(np.complex64)}
        )
    )

    # Case 8: 3D array, axis=2, uint8
    arr = np.random.randint(0, 256, size=(3, 2, 2), dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arr, "axis": 2, "dtype": np.dtype(np.uint8)})
    )

    # Case 9: 3D array, negative axis=-3, int32
    arr = np.random.randint(-10, 10, size=(4, 2, 3), dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arr, "axis": -3, "dtype": np.dtype(np.int32)})
    )

    # Case 10: 4D array, axis=3, float32
    arr = np.random.randn(2, 3, 2, 2).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arr, "axis": 3, "dtype": np.dtype(np.float32)})
    )

    # Case 11: 4D array, axis=-4, float64
    arr = np.random.randn(3, 2, 4, 2).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"arrays": arr, "axis": -4, "dtype": np.dtype(np.float64)}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.stack_1"] = stack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.stack_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.stack_1'.")


check_valid('jax.numpy.stack', generated_inputs['jax.numpy.stack_1'], lib="jax", suffix=1)
