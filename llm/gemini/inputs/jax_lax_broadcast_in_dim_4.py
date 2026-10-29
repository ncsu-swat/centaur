
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_in_dim_inputs():
    list_of_inputs = []

    # Input 1: Scalar to 1D
    operand = np.array(5.0, dtype=np.float32)
    shape = [4]
    broadcast_dimensions = []
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 2: 1D to 2D (row vector broadcast)
    operand = np.array([1, 2, 3], dtype=np.int32)
    shape = [4, 3]
    broadcast_dimensions = [1]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 3: 1D to 2D (column vector broadcast)
    operand = np.array([1.5, -2.5], dtype=np.float64)
    shape = [2, 5]
    broadcast_dimensions = [0]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 4: 2D to 3D with transpose
    operand = np.ones((3, 4), dtype=np.int64)
    shape = [4, 2, 3]
    broadcast_dimensions = [2, 0]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 5: 2D with singleton dimension expansion, boolean dtype
    operand = np.array([[True], [False]], dtype=np.bool_)
    shape = [2, 7]
    broadcast_dimensions = [0, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 6: 3D to 4D complex array with expansion
    operand = np.zeros((2, 1, 4), dtype=np.complex64)
    shape = [5, 2, 3, 4]
    broadcast_dimensions = [1, 2, 3]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 7: 0D scalar to 3D int16 array
    operand = np.array(-42, dtype=np.int16)
    shape = [2, 3, 4]
    broadcast_dimensions = []
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 8: 1D uint8 array to 4D
    operand = np.arange(6, dtype=np.uint8)
    shape = [2, 1, 6, 3]
    broadcast_dimensions = [2]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 9: 2D float16 array with negative values
    operand = np.array([[-1.0, -2.0, -3.0]], dtype=np.float16)
    shape = [5, 3]
    broadcast_dimensions = [0, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 10: 3D float32 array with reverse dimension ordering (transpose-like)
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    shape = [4, 3, 2]
    broadcast_dimensions = [2, 1, 0]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 11: 4D to 5D array with multiple broadcasting singleton dimensions
    operand = np.ones((1, 2, 1, 3), dtype=np.int32)
    shape = [2, 4, 2, 5, 3]
    broadcast_dimensions = [1, 2, 3, 4]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.lax.broadcast_in_dim_4"] = broadcast_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.broadcast_in_dim_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.broadcast_in_dim_4'.")


check_valid('jax.lax.broadcast_in_dim', generated_inputs['jax.lax.broadcast_in_dim_4'], lib="jax", suffix=4)
