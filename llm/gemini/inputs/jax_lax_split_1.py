
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_split_inputs():
    list_of_inputs = []

    # Input 1: 1D array, split into two uneven chunks
    operand = np.arange(10, dtype=np.float32)
    sizes = [4, 6]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 2: 2D array, split along rows (axis 0)
    operand = np.random.randn(6, 4).astype(np.float32)
    sizes = [2, 2, 2]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 3: 2D array, split along columns (axis 1)
    operand = np.random.randn(3, 8).astype(np.float64)
    sizes = [3, 5]
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 4: 3D array with negative axis
    operand = np.random.randint(-10, 10, size=(2, 3, 7), dtype=np.int32)
    sizes = [2, 2, 3]
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 5: 3D array, split along middle axis
    operand = np.ones((4, 5, 2), dtype=np.int64)
    sizes = [1, 4]
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 6: 4D array, float32, split along axis 0
    operand = np.zeros((4, 2, 2, 2), dtype=np.float32)
    sizes = [1, 1, 1, 1]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 7: 1D boolean array with zero-sized splits
    operand = np.array([True, False, True, True], dtype=np.bool_)
    sizes = [0, 4, 0]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 8: 2D array, negative axis -2
    operand = np.arange(12, dtype=np.float32).reshape(4, 3)
    sizes = [1, 2, 1]
    axis = -2
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 9: 5D array, complex64, split along axis 2
    operand = (
        np.random.randn(2, 2, 6, 2, 2) + 1j * np.random.randn(2, 2, 6, 2, 2)
    ).astype(np.complex64)
    sizes = [1, 2, 3]
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 10: 1D array, single split returning full array
    operand = np.linspace(-5.0, 5.0, 8, dtype=np.float64)
    sizes = [8]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    # Input 11: 3D uint8 array, split along axis 2
    operand = np.full((3, 2, 4), 255, dtype=np.uint8)
    sizes = [2, 2]
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "sizes": sizes, "axis": axis})
    )

    return list_of_inputs


generated_inputs["jax.lax.split_1"] = lax_split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.split_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.split_1'.")


check_valid('jax.lax.split', generated_inputs['jax.lax.split_1'], lib="jax", suffix=1)
