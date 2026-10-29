
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def broadcast_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar to 1D array (float32)
    operand = np.array(3.14, dtype=np.float32)
    shape = (5,)
    broadcast_dimensions = []
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    }))

    # Input 2: 1D to 2D row broadcast (int32)
    operand = np.array([1, 2, 3], dtype=np.int32)
    shape = (4, 3)
    broadcast_dimensions = [1]
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    }))

    # Input 3: 1D to 2D column broadcast (int64)
    operand = np.array([10, 20, 30, 40], dtype=np.int64)
    shape = (4, 5)
    broadcast_dimensions = [0]
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    }))

    # Input 4: 2D with singleton dimension expanded to 3D (float32)
    operand = np.arange(3, dtype=np.float32).reshape(3, 1)
    shape = (2, 3, 4)
    broadcast_dimensions = [1, 2]
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    }))

    # Input 5: 2D transpose broadcast (bool)
    operand = np.array([[True, False], [False, True], [True, True]], dtype=np.bool_)
    shape = (2, 3)
    broadcast_dimensions = [1, 0]
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    }))

    # Input 6: 1D singleton broadcast into 3D (uint8)
    operand = np.array([42], dtype=np.uint8)
    shape = (2, 3, 4)
    broadcast_dimensions = [1]
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    }))

    # Input 7: 3D to 4D broadcasting with singleton dimension (float64)
    operand = np.random.randn(2, 1, 4).astype(np.float64)
    shape = (3, 2, 5, 4)
    broadcast_dimensions = [1, 2, 3]
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    }))

    # Input 8: 0D negative float scalar to 4D tensor (float16)
    operand = np.array(-2.5, dtype=np.float16)
    shape = (2, 2, 2, 2)
    broadcast_dimensions = []
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    }))

    # Input 9: 2D complex numbers with dimension expansion (complex64)
    operand = np.array([[1.0 + 2.0j, -3.0 + 4.0j]], dtype=np.complex64)
    shape = (6, 2)
    broadcast_dimensions = [0, 1]
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    }))

    # Input 10: 3D permutation/transposition of dimensions (int16)
    operand = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    shape = (4, 2, 3)
    broadcast_dimensions = [1, 2, 0]
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    }))

    # Input 11: 1D with negative integers to 3D (int8)
    operand = np.array([-10, -20, -30], dtype=np.int8)
    shape = (1, 3, 1)
    broadcast_dimensions = [1]
    list_of_inputs.append(copy.deepcopy({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    }))

    return list_of_inputs

generated_inputs["jax.lax.broadcast_in_dim_2"] = broadcast_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.broadcast_in_dim_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.broadcast_in_dim_2'.")


check_valid('jax.lax.broadcast_in_dim', generated_inputs['jax.lax.broadcast_in_dim_2'], lib="jax", suffix=2)
