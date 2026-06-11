
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_broadcast_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D to 2D broadcast
    operand = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = [2, 3]
    broadcast_dimensions = (1,)
    list_of_inputs.append({"operand": operand, "shape": shape, "broadcast_dimensions": broadcast_dimensions})

    # Input 2: 2D to 3D broadcast (prepended dimension)
    operand = np.random.randn(2, 3).astype(np.float32)
    shape = [2, 3, 4]
    broadcast_dimensions = (0, 1)
    list_of_inputs.append({"operand": operand, "shape": shape, "broadcast_dimensions": broadcast_dimensions})

    # Input 3: Implicit transpose and broadcast
    operand = np.random.randn(3, 1).astype(np.float64)
    shape = [2, 3, 4]
    broadcast_dimensions = (1, 0)
    list_of_inputs.append({"operand": operand, "shape": shape, "broadcast_dimensions": broadcast_dimensions})

    # Input 4: Transpose via broadcast_in_dim
    operand = np.arange(6, dtype=np.int32).reshape(2, 3)
    shape = [3, 2]
    broadcast_dimensions = (1, 0)
    list_of_inputs.append({"operand": operand, "shape": shape, "broadcast_dimensions": broadcast_dimensions})

    # Input 5: Scalar to 1D
    operand = np.array(5, dtype=np.int64)
    shape = [10]
    broadcast_dimensions = ()
    list_of_inputs.append({"operand": operand, "shape": shape, "broadcast_dimensions": broadcast_dimensions})

    # Input 6: 3D with dimensions of size 1, boolean type
    operand = np.random.randint(0, 2, size=(1, 5, 1)).astype(np.bool_)
    shape = [3, 5, 4]
    broadcast_dimensions = (0, 1, 2)
    list_of_inputs.append({"operand": operand, "shape": shape, "broadcast_dimensions": broadcast_dimensions})

    # Input 7: 1D to 3D (broadcasting to the middle dimension)
    operand = np.random.randn(5).astype(np.float32)
    shape = [5, 5, 5]
    broadcast_dimensions = (1,)
    list_of_inputs.append({"operand": operand, "shape": shape, "broadcast_dimensions": broadcast_dimensions})

    # Input 8: 2D to 3D with skipped dimension
    operand = np.ones((2, 2), dtype=np.float64)
    shape = [2, 3, 2]
    broadcast_dimensions = (0, 2)
    list_of_inputs.append({"operand": operand, "shape": shape, "broadcast_dimensions": broadcast_dimensions})

    # Input 9: 3D to 4D transpose and broadcast
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    shape = [4, 3, 2, 5]
    broadcast_dimensions = (2, 1, 0)
    list_of_inputs.append({"operand": operand, "shape": shape, "broadcast_dimensions": broadcast_dimensions})

    # Input 10: 2D size 1 to 2D larger shape
    operand = np.array([[1.0]], dtype=np.float32)
    shape = [5, 5]
    broadcast_dimensions = (0, 1)
    list_of_inputs.append({"operand": operand, "shape": shape, "broadcast_dimensions": broadcast_dimensions})

    return list_of_inputs

generated_inputs["jax.lax.broadcast_in_dim_3"] = jax_lax_broadcast_in_dim_inputs()

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
