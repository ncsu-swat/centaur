
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar to 3D tensor
    operand = np.array(5.0, dtype=np.float32)
    shape = (2, 3, 4)
    broadcast_dimensions = ()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 2: 1D to 2D row broadcast
    operand = np.array([1, -2, 3], dtype=np.int32)
    shape = (4, 3)
    broadcast_dimensions = (1,)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 3: 1D to 2D column broadcast
    operand = np.array([10, -20, 30, 40], dtype=np.int64)
    shape = (4, 5)
    broadcast_dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 4: 2D to 3D standard broadcast
    operand = np.random.randn(3, 4).astype(np.float32)
    shape = (2, 3, 4)
    broadcast_dimensions = (1, 2)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 5: 2D to 3D with dimension permutation (implicit transpose)
    operand = np.random.randn(3, 2).astype(np.float64)
    shape = (2, 3, 5)
    broadcast_dimensions = (1, 0)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 6: 2D with singleton dimension expansion
    operand = np.random.randn(1, 4).astype(np.float32)
    shape = (5, 4)
    broadcast_dimensions = (0, 1)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 7: Boolean array broadcast to 4D
    operand = np.array([[True, False], [False, True]], dtype=np.bool_)
    shape = (3, 2, 2, 4)
    broadcast_dimensions = (1, 2)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 8: 3D to 5D with singleton dimension
    operand = np.ones((2, 1, 3), dtype=np.int16)
    shape = (4, 2, 5, 3, 6)
    broadcast_dimensions = (1, 2, 3)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 9: Complex numbers 1D to 2D
    operand = np.array([-1.5 + 2.0j, 3.0 - 4.5j], dtype=np.complex64)
    shape = (2, 2)
    broadcast_dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 10: 1D identity broadcast (same shape)
    operand = np.array([-10, 0, 10], dtype=np.int8)
    shape = (3,)
    broadcast_dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 11: 4D to 4D full reverse transposition
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    shape = (5, 4, 3, 2)
    broadcast_dimensions = (3, 2, 1, 0)
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


generated_inputs["jax.lax.broadcast_in_dim_1"] = broadcast_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.broadcast_in_dim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.broadcast_in_dim_1'.")


check_valid('jax.lax.broadcast_in_dim', generated_inputs['jax.lax.broadcast_in_dim_1'], lib="jax", suffix=1)
