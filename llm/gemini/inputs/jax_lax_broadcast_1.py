
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def broadcast_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array (float32)
    operand = np.array(3.14, dtype=np.float32)
    sizes = (2, 3)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 2: 1D array with negative numbers (int32)
    operand = np.array([-1, 0, 5, -10], dtype=np.int32)
    sizes = (4,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 3: 2D array (float64)
    operand = np.random.randn(3, 4).astype(np.float64)
    sizes = (2, 5)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 4: 3D boolean array
    operand = np.array(
        [[[True, False], [False, True]], [[True, True], [False, False]]],
        dtype=np.bool_,
    )
    sizes = (1, 2)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 5: 1D complex numbers array
    operand = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64)
    sizes = (3, 2, 2)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 6: 2D array with empty sizes tuple (identity broadcast)
    operand = np.ones((2, 3), dtype=np.int64)
    sizes = ()
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 7: 4D uint8 array
    operand = np.random.randint(0, 255, size=(1, 2, 3, 4), dtype=np.uint8)
    sizes = (2,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 8: 0D int16 array with 3D leading expansion
    operand = np.array(-42, dtype=np.int16)
    sizes = (1, 1, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 9: 1D zero-size array
    operand = np.array([], dtype=np.float32)
    sizes = (3, 2)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 10: 2D float16 array
    operand = np.random.randn(5, 2).astype(np.float16)
    sizes = (6,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    # Input 11: 3D int8 array with negative elements
    operand = np.random.randint(-128, 127, size=(2, 2, 2), dtype=np.int8)
    sizes = (2, 3)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes})
    )

    return list_of_inputs


generated_inputs["jax.lax.broadcast_1"] = broadcast_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.broadcast_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.broadcast_1'.")


check_valid('jax.lax.broadcast', generated_inputs['jax.lax.broadcast_1'], lib="jax", suffix=1)
