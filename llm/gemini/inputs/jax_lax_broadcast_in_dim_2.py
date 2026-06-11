
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def broadcast_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D broadcasting
    operand = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = (2, 3)
    broadcast_dimensions = [1]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 2: 1D to 3D broadcasting
    operand = np.array([10, 20, 30, 40], dtype=np.int32)
    shape = (2, 4, 3)
    broadcast_dimensions = [1]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 3: Scalar (0D) to 2D broadcasting
    operand = np.array(5.5, dtype=np.float64)
    shape = (5, 5)
    broadcast_dimensions = []
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 4: 2D to 3D with implicit transpose (as in the example)
    operand = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    shape = (2, 3, 4)
    broadcast_dimensions = [1, 0]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 5: Simple 2D to 2D broadcasting
    operand = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32) # shape (1, 5)
    shape = (4, 5)
    broadcast_dimensions = [0, 1]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 6: Boolean array broadcasting
    operand = np.array([[True, False], [False, True]], dtype=np.bool_)
    shape = (2, 2, 3)
    broadcast_dimensions = [0, 1]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 7: 3D to 4D broadcasting with singleton dimension
    operand = np.random.randn(2, 1, 3).astype(np.float32)
    shape = (2, 5, 3, 4)
    broadcast_dimensions = [0, 1, 2]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 8: Negative numbers in operand (1D to 2D)
    operand = np.array([-1.5, -2.5], dtype=np.float32)
    shape = (2, 2)
    broadcast_dimensions = [0]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 9: Complex numbers broadcasting
    operand = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    shape = (3, 3)
    broadcast_dimensions = [0]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

    # Input 10: 2D to 4D broadcasting mapping to non-consecutive dimensions
    operand = np.arange(6, dtype=np.int16).reshape(2, 3)
    shape = (4, 2, 5, 3)
    broadcast_dimensions = [1, 3]
    list_of_inputs.append({
        "operand": operand,
        "shape": shape,
        "broadcast_dimensions": broadcast_dimensions
    })

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
