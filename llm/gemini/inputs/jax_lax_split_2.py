
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def split_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, split into 2 chunks along axis 0
    operand = np.random.randn(6).astype(np.float32)
    sizes = (2, 4)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 2: 2D float64 array, split along axis 0
    operand = np.random.randn(8, 4).astype(np.float64)
    sizes = (3, 5)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 3: 2D int32 array, split along axis 1
    operand = np.random.randint(-10, 10, size=(5, 9)).astype(np.int32)
    sizes = (3, 3, 3)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 4: 2D array, negative axis
    operand = np.random.randn(4, 7).astype(np.float32)
    sizes = (2, 5)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 5: 3D float32 array, split along axis 1
    operand = np.random.randn(2, 6, 3).astype(np.float32)
    sizes = (1, 2, 3)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 6: 3D int64 array, negative axis
    operand = np.random.randint(0, 100, size=(3, 4, 8)).astype(np.int64)
    sizes = (4, 4)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 7: 4D float32 array, split along axis 2
    operand = np.random.randn(2, 3, 5, 4).astype(np.float32)
    sizes = (1, 4)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 8: 1D bool array, single split (keeps full size)
    operand = np.array([True, False, True, True, False], dtype=np.bool_)
    sizes = (5,)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 9: 2D complex64 array, split with empty chunk (size 0)
    operand = (
        np.random.randn(4, 6) + 1j * np.random.randn(4, 6)
    ).astype(np.complex64)
    sizes = (0, 4)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 10: 5D uint8 array, split along the first axis using negative index
    operand = np.random.randint(0, 255, size=(3, 2, 2, 2, 2)).astype(np.uint8)
    sizes = (1, 2)
    axis = -5
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 11: 3D float32 array, split into 4 chunks along axis 2
    operand = np.random.randn(2, 2, 10).astype(np.float32)
    sizes = (1, 2, 3, 4)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    return list_of_inputs


generated_inputs["jax.lax.split_2"] = split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.split_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.split_2'.")


check_valid('jax.lax.split', generated_inputs['jax.lax.split_2'], lib="jax", suffix=2)
