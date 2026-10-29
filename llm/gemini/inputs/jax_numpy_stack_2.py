
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def stack_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0, float32
    arrays = np.random.randn(5).astype(np.float32)
    axis = 0
    dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "axis": axis, "dtype": dtype})
    )

    # Input 2: 1D array, axis=-1, int32
    arrays = np.random.randint(-10, 10, size=(4,), dtype=np.int32)
    axis = -1
    dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "axis": axis, "dtype": dtype})
    )

    # Input 3: 2D array, axis=0, float64
    arrays = np.random.randn(3, 4).astype(np.float64)
    axis = 0
    dtype = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "axis": axis, "dtype": dtype})
    )

    # Input 4: 2D array, axis=1, int64
    arrays = np.random.randint(0, 100, size=(5, 6), dtype=np.int64)
    axis = 1
    dtype = np.dtype(np.int64)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "axis": axis, "dtype": dtype})
    )

    # Input 5: 2D array, axis=-1, bool
    arrays = np.random.choice([True, False], size=(3, 2))
    axis = -1
    dtype = np.dtype(np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "axis": axis, "dtype": dtype})
    )

    # Input 6: 2D array, axis=-2, float16
    arrays = np.random.randn(4, 3).astype(np.float16)
    axis = -2
    dtype = np.dtype(np.float16)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "axis": axis, "dtype": dtype})
    )

    # Input 7: 3D array, axis=0, complex64
    arrays = (
        np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
    ).astype(np.complex64)
    axis = 0
    dtype = np.dtype(np.complex64)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "axis": axis, "dtype": dtype})
    )

    # Input 8: 3D array, axis=2, uint8
    arrays = np.random.randint(0, 255, size=(2, 4, 3), dtype=np.uint8)
    axis = 2
    dtype = np.dtype(np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "axis": axis, "dtype": dtype})
    )

    # Input 9: 3D array, axis=-3, int16
    arrays = np.random.randint(-100, 100, size=(3, 2, 2), dtype=np.int16)
    axis = -3
    dtype = np.dtype(np.int16)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "axis": axis, "dtype": dtype})
    )

    # Input 10: 4D array, axis=1, float32
    arrays = np.random.randn(2, 2, 3, 4).astype(np.float32)
    axis = 1
    dtype = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "axis": axis, "dtype": dtype})
    )

    # Input 11: 4D array, axis=-1, float64
    arrays = np.random.randn(3, 1, 2, 2).astype(np.float64)
    axis = -1
    dtype = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"arrays": arrays, "axis": axis, "dtype": dtype})
    )

    return list_of_inputs


generated_inputs["jax.numpy.stack_2"] = stack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.stack_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.stack_2'.")


check_valid('jax.numpy.stack', generated_inputs['jax.numpy.stack_2'], lib="jax", suffix=2)
