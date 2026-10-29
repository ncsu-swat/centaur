
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_in_dim_inputs():
    list_of_inputs = []

    # Input 1: Scalar float32 to 1D array
    operand = np.array(3.14, dtype=np.float32)
    shape = [5]
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

    # Input 2: Scalar int32 to 3D array
    operand = np.array(-7, dtype=np.int32)
    shape = [2, 3, 4]
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

    # Input 3: 1D array to 1D array (same size)
    operand = np.array([1.0, -2.5, 3.2], dtype=np.float64)
    shape = [3]
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

    # Input 4: 1D array broadcasted along column dimension of 2D
    operand = np.array([10, 20, 30, 40], dtype=np.int64)
    shape = [3, 4]
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

    # Input 5: 1D array broadcasted along row dimension of 2D
    operand = np.array([True, False, True], dtype=np.bool_)
    shape = [3, 5]
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

    # Input 6: 2D array broadcasted to 3D without transpose
    operand = np.arange(6, dtype=np.float32).reshape(2, 3)
    shape = [4, 2, 3]
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

    # Input 7: 2D array broadcasted to 3D with transpose
    operand = np.ones((4, 2), dtype=np.float32)
    shape = [2, 3, 4]
    broadcast_dimensions = (2, 0)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 8: 2D array with singleton dimension expanded
    operand = np.array([[1.0 + 2.0j], [-3.0 + 0.5j]], dtype=np.complex64)
    shape = [2, 5]
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

    # Input 9: 3D array broadcasted to 4D with dimension reordering
    operand = np.random.randn(2, 3, 1).astype(np.float32)
    shape = [5, 2, 4, 3]
    broadcast_dimensions = (1, 3, 2)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 10: 1D singleton array broadcasted to 4D
    operand = np.array([42], dtype=np.int16)
    shape = [2, 3, 4, 5]
    broadcast_dimensions = (2,)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "shape": shape,
                "broadcast_dimensions": broadcast_dimensions,
            }
        )
    )

    # Input 11: 3D to 3D permutation/transposition
    operand = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    shape = [4, 2, 3]
    broadcast_dimensions = (1, 2, 0)
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


generated_inputs["jax.lax.broadcast_in_dim_3"] = broadcast_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.broadcast_in_dim_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.broadcast_in_dim_3'.")


check_valid('jax.lax.broadcast_in_dim', generated_inputs['jax.lax.broadcast_in_dim_3'], lib="jax", suffix=3)
