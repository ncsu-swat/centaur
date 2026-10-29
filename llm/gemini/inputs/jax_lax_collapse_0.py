
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def collapse_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 array, collapse first two dimensions
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 0, "stop_dimension": 2}
    )

    # Input 2: 3D int32 array, collapse last two dimensions
    operand = np.random.randint(-10, 10, size=(4, 5, 6), dtype=np.int32)
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 1, "stop_dimension": 3}
    )

    # Input 3: 4D float64 array, collapse middle two dimensions
    operand = np.random.randn(2, 3, 4, 5).astype(np.float64)
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 1, "stop_dimension": 3}
    )

    # Input 4: 4D bool array, collapse all dimensions into 1D
    operand = np.random.choice([True, False], size=(2, 2, 2, 2))
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 0, "stop_dimension": 4}
    )

    # Input 5: 2D uint8 array, collapse both dimensions
    operand = np.random.randint(0, 255, size=(10, 20), dtype=np.uint8)
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 0, "stop_dimension": 2}
    )

    # Input 6: 5D int64 array, collapse dimensions 1 through 4
    operand = np.random.randint(-100, 100, size=(2, 3, 2, 4, 5), dtype=np.int64)
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 1, "stop_dimension": 4}
    )

    # Input 7: 1D float32 array, collapse the single dimension
    operand = np.random.randn(16).astype(np.float32)
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 0, "stop_dimension": 1}
    )

    # Input 8: 3D complex64 array, collapse single dimension (identity operation)
    real_part = np.random.randn(3, 4, 5).astype(np.float32)
    imag_part = np.random.randn(3, 4, 5).astype(np.float32)
    operand = (real_part + 1j * imag_part).astype(np.complex64)
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 1, "stop_dimension": 2}
    )

    # Input 9: 4D float16 array, collapse first three dimensions
    operand = np.random.randn(3, 2, 4, 2).astype(np.float16)
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 0, "stop_dimension": 3}
    )

    # Input 10: 3D int16 array with unit dimensions, collapse all
    operand = np.ones((1, 5, 1), dtype=np.int16)
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 0, "stop_dimension": 3}
    )

    # Input 11: 4D float32 array, collapse empty range at dimension 2 (inserts size 1 dim)
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 2, "stop_dimension": 2}
    )

    # Input 12: 5D float32 array, collapse last three dimensions
    operand = np.random.randn(2, 3, 2, 2, 2).astype(np.float32)
    list_of_inputs.append(
        {"operand": operand, "start_dimension": 2, "stop_dimension": 5}
    )

    return [copy.deepcopy(x) for x in list_of_inputs]


generated_inputs["jax.lax.collapse"] = collapse_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.collapse' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.collapse'.")


check_valid('jax.lax.collapse', generated_inputs['jax.lax.collapse'], lib="jax", suffix=0)
