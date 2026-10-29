
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def shift_right_arithmetic_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays
    x = np.array([-16, -8, -4, 0, 4, 8, 16], dtype=np.int32)
    y = np.array([1, 2, 1, 0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D int64 arrays with negative numbers
    x = np.array([[-100, -50], [25, 100]], dtype=np.int64)
    y = np.array([[2, 1], [0, 3]], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 0D scalar-like int32 arrays
    x = np.array(-32, dtype=np.int32)
    y = np.array(2, dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 1D int32 arrays
    x = np.array([1024, 512, 256, 128], dtype=np.int32)
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 2D int32 arrays with broadcasting (same ndim)
    x = np.array([[-128], [64], [-32]], dtype=np.int32)
    y = np.array([[1, 2, 3, 4]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 3D int32 arrays
    x = np.array(
        [[[-8, 16], [-32, 64]], [[-128, 256], [-512, 1024]]], dtype=np.int32
    )
    y = np.ones((2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 4D int32 arrays with broadcasting (same ndim)
    x = np.random.randint(-1000, 1000, size=(1, 2, 1, 3), dtype=np.int32)
    y = np.random.randint(0, 5, size=(2, 2, 4, 3), dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: 1D int64 arrays with larger shifts
    x = np.array([-(2**60), 2**60, -1, 1], dtype=np.int64)
    y = np.array([60, 30, 10, 0], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: 2D int64 arrays
    x = np.array([[255, -128], [64, -32]], dtype=np.int64)
    y = np.array([[1, 2], [3, 4]], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 1D int64 arrays
    x = np.array([65535, -32768, 16384, -8192], dtype=np.int64)
    y = np.array([4, 4, 4, 4], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 11: 3D int32 arrays with zeros
    x = np.zeros((3, 3, 3), dtype=np.int32)
    y = np.random.randint(0, 10, size=(3, 3, 3), dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs


generated_inputs["jax.lax.shift_right_arithmetic"] = (
    shift_right_arithmetic_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.shift_right_arithmetic' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.shift_right_arithmetic'.")


check_valid('jax.lax.shift_right_arithmetic', generated_inputs['jax.lax.shift_right_arithmetic'], lib="jax", suffix=0)
