
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def broadcast_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D array to 2D array
    operand = np.arange(3, dtype=np.float32)
    shape = [2, 3]
    broadcast_dimensions = [1]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 2: Scalar to 3D array
    operand = np.array(5.0, dtype=np.float32)
    shape = [2, 3, 4]
    broadcast_dimensions = []
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 3: 2D array to 3D array (no transpose, match dimensions)
    operand = np.random.randn(3, 4).astype(np.float32)
    shape = [2, 3, 4]
    broadcast_dimensions = [1, 2]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 4: 2D array to 3D array with implicit transpose
    operand = np.random.randn(4, 3).astype(np.float32)
    shape = [2, 3, 4]
    broadcast_dimensions = [2, 1]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 5: 2D array with size-1 dimension to be broadcasted
    operand = np.random.randn(1, 5).astype(np.float32)
    shape = [3, 5]
    broadcast_dimensions = [0, 1]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 6: 1D integer array to 3D array
    operand = np.array([10, 20], dtype=np.int32)
    shape = [2, 5, 5]
    broadcast_dimensions = [0]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 7: 2D boolean array to 4D array with broadcasting
    operand = np.array([[True, False]], dtype=np.bool_)
    shape = [2, 3, 2, 4]
    broadcast_dimensions = [1, 2]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 8: 3D double array to 5D array (complex mappings)
    operand = np.random.randn(2, 1, 4).astype(np.float64)
    shape = [2, 3, 4, 5, 6]
    broadcast_dimensions = [0, 3, 2]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 9: 1D array to 1D array (identity broadcast)
    operand = np.array([1, 2, 3], dtype=np.int64)
    shape = [3]
    broadcast_dimensions = [0]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 10: Size-1 dimensions broadcast to larger shapes
    operand = np.ones((1, 1), dtype=np.float32)
    shape = [10, 20, 30]
    broadcast_dimensions = [1, 2]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

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
