
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def broadcast_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D (adding leading dimension)
    operand = np.random.randn(3).astype(np.float32)
    shape = (2, 3)
    broadcast_dimensions = (1,)
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 2: 1D to 2D (adding trailing dimension)
    operand = np.random.randn(3).astype(np.float32)
    shape = (3, 4)
    broadcast_dimensions = (0,)
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 3: 2D to 3D (standard broadcasting)
    operand = np.random.randn(3, 1).astype(np.float32)
    shape = (2, 3, 4)
    broadcast_dimensions = (1, 2)
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 4: 2D to 3D with implicit transpose
    operand = np.random.randn(3, 1).astype(np.float32)
    shape = (2, 3, 4)
    broadcast_dimensions = (1, 0)
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 5: Scalar to 3D
    operand = np.array(4.2, dtype=np.float64)
    shape = (2, 3, 4)
    broadcast_dimensions = ()
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 6: No-op broadcasting (same shape)
    operand = np.random.randn(5, 5).astype(np.float64)
    shape = (5, 5)
    broadcast_dimensions = (0, 1)
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 7: 1D to 3D with integer types
    operand = np.arange(4, dtype=np.int32)
    shape = (2, 3, 4)
    broadcast_dimensions = (2,)
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 8: 3D to 3D (broadcasting size-1 dimensions)
    operand = np.random.randn(1, 5, 1).astype(np.float32)
    shape = (2, 5, 3)
    broadcast_dimensions = (0, 1, 2)
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 9: 2D to 4D with transposition (float16)
    operand = np.random.randn(4, 5).astype(np.float16)
    shape = (2, 5, 3, 4)
    broadcast_dimensions = (3, 1)
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 10: 1D to 3D with large dimensions (int64)
    operand = np.arange(100, dtype=np.int64)
    shape = (10, 100, 5)
    broadcast_dimensions = (1,)
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 11: Negative values in operand
    operand = np.array([[-1.5, -2.0, -3.5], [-4.0, -5.5, -6.0]], dtype=np.float32)
    shape = (2, 3, 4)
    broadcast_dimensions = (0, 1)
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

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
