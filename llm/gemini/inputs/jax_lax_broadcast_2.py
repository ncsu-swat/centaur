
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, add two leading dimensions
    operand = np.random.randn(4, 5).astype(np.float32)
    sizes = [2, 3]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 2: 1D int32 array with negative values, add one leading dimension
    operand = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    sizes = [4]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 3: Scalar (0D) float64 array, add multiple leading dimensions
    operand = np.array(3.14159, dtype=np.float64)
    sizes = [2, 3, 4]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 4: 3D float32 array, add empty sizes (no-op broadcast)
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    sizes = []
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 5: Boolean array, add leading dimensions of size 1
    operand = np.array([[True, False], [False, True]], dtype=np.bool_)
    sizes = [1, 1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 6: Complex64 array with negative real and imaginary parts
    operand = np.array(
        [-1.0 + 2.0j, 3.0 - 4.0j, -5.0 - 6.0j], dtype=np.complex64
    )
    sizes = [3, 2]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 7: uint8 array, add a single leading dimension
    operand = np.arange(12, dtype=np.uint8).reshape(3, 4)
    sizes = [5]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 8: int64 array, add zero-sized leading dimension
    operand = np.array([1, 2, 3], dtype=np.int64)
    sizes = [0, 2]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 9: 4D float32 array, add two leading dimensions
    operand = np.random.randn(1, 2, 1, 3).astype(np.float32)
    sizes = [2, 2]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 10: 1D float64 array with negative values, add 4 leading dimensions
    operand = np.array([-2.5, -0.5, 1.5], dtype=np.float64)
    sizes = [2, 1, 3, 1]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 11: 0D int32 array, add a single leading dimension
    operand = np.array(-42, dtype=np.int32)
    sizes = [7]
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    return list_of_inputs


generated_inputs["jax.lax.broadcast_2"] = broadcast_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.broadcast_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.broadcast_2'.")


check_valid('jax.lax.broadcast', generated_inputs['jax.lax.broadcast_2'], lib="jax", suffix=2)
