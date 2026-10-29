
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def empty_like_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 prototype, int32 dtype, 2D shape
    list_of_inputs.append(
        {
            "prototype": np.zeros((3, 3), dtype=np.float32),
            "dtype": np.dtype(np.int32),
            "shape": (4, 4),
        }
    )

    # Input 2: 3D int32 prototype, float32 dtype, 2D shape
    list_of_inputs.append(
        {
            "prototype": np.ones((2, 2, 2), dtype=np.int32),
            "dtype": np.dtype(np.float32),
            "shape": (3, 5),
        }
    )

    # Input 3: 1D int32 prototype, bool dtype, 1D shape
    list_of_inputs.append(
        {
            "prototype": np.array([1, 2, 3], dtype=np.int32),
            "dtype": np.dtype(np.bool_),
            "shape": (6,),
        }
    )

    # Input 4: 0D float32 prototype (scalar array), float32 dtype, 0D shape
    list_of_inputs.append(
        {
            "prototype": np.array(-5.5, dtype=np.float32),
            "dtype": np.dtype(np.float32),
            "shape": (),
        }
    )

    # Input 5: 4D float32 prototype, float32 dtype, 3D shape
    list_of_inputs.append(
        {
            "prototype": np.random.randn(2, 3, 4, 5).astype(np.float32),
            "dtype": np.dtype(np.float32),
            "shape": (1, 2, 3),
        }
    )

    # Input 6: 2D bool prototype, int32 dtype, 3D shape
    list_of_inputs.append(
        {
            "prototype": np.array([[True, False], [False, True]], dtype=np.bool_),
            "dtype": np.dtype(np.int32),
            "shape": (2, 2, 2),
        }
    )

    # Input 7: 2D empty-dimension prototype, int32 dtype, zero-dimension shape
    list_of_inputs.append(
        {
            "prototype": np.zeros((0, 5), dtype=np.float32),
            "dtype": np.dtype(np.int32),
            "shape": (0, 2),
        }
    )

    # Input 8: 1D negative integer prototype, float32 dtype, 2D shape
    list_of_inputs.append(
        {
            "prototype": np.array([-10, -5, 0, 5, 10], dtype=np.int32),
            "dtype": np.dtype(np.float32),
            "shape": (5, 1),
        }
    )

    # Input 9: 3D float32 prototype, bool dtype, 4D shape
    list_of_inputs.append(
        {
            "prototype": np.ones((1, 1, 1), dtype=np.float32),
            "dtype": np.dtype(np.bool_),
            "shape": (2, 3, 4, 2),
        }
    )

    # Input 10: 1D int32 prototype, int32 dtype, 2D shape
    list_of_inputs.append(
        {
            "prototype": np.zeros((4,), dtype=np.int32),
            "dtype": np.dtype(np.int32),
            "shape": (2, 2),
        }
    )

    # Input 11: 1D bool prototype, bool dtype, 1D shape
    list_of_inputs.append(
        {
            "prototype": np.array([True, False, True], dtype=np.bool_),
            "dtype": np.dtype(np.bool_),
            "shape": (5,),
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.empty_like_1"] = empty_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.empty_like_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.empty_like_1'.")


check_valid('jax.numpy.empty_like', generated_inputs['jax.numpy.empty_like_1'], lib="jax", suffix=1)
