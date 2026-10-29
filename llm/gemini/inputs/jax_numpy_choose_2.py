
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def choose_inputs():
    list_of_inputs = []

    # Input 1: 1D indices, 2D choices, mode='raise'
    a = np.array([0, 1, 2, 1], dtype=np.int32)
    choices = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int32
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "raise"})
    )

    # Input 2: 1D indices with out-of-bounds, mode='clip'
    a = np.array([-2, 0, 1, 5], dtype=np.int64)
    choices = np.array(
        [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]], dtype=np.float32
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "clip"})
    )

    # Input 3: 1D indices with negative / wrapping values, mode='wrap'
    a = np.array([-1, 3, -4, 2], dtype=np.int32)
    choices = np.array(
        [
            [1.5, 2.5, 3.5, 4.5],
            [5.5, 6.5, 7.5, 8.5],
            [9.5, 10.5, 11.5, 12.5],
        ],
        dtype=np.float64,
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "wrap"})
    )

    # Input 4: 2D indices, 3D choices, mode='raise'
    a = np.array([[0, 1], [1, 0]], dtype=np.int32)
    choices = np.zeros((2, 2, 2), dtype=np.int16)
    choices[0] = [[1, 2], [3, 4]]
    choices[1] = [[5, 6], [7, 8]]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "raise"})
    )

    # Input 5: Scalar-like broadcasted choices with 2D indices, mode='clip'
    a = np.array([[0, 2, 10], [-1, 1, 0]], dtype=np.int32)
    choices = np.array(
        [[[1, 1, 1]], [[2, 2, 2]], [[3, 3, 3]]], dtype=np.float32
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "clip"})
    )

    # Input 6: 3D indices and 4D choices, mode='wrap'
    a = np.random.randint(-5, 5, size=(2, 3, 4)).astype(np.int32)
    choices = np.random.randn(4, 2, 3, 4).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "wrap"})
    )

    # Input 7: Boolean choices, mode='raise'
    a = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    choices = np.array(
        [
            [True, False, True, False, False],
            [False, True, False, True, True],
        ],
        dtype=bool,
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "raise"})
    )

    # Input 8: Complex choices, mode='clip'
    a = np.array([3, -1, 0, 2], dtype=np.int64)
    choices = np.array(
        [
            [1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j],
            [5 + 6j, 6 + 7j, 7 + 8j, 8 + 9j],
            [9 + 10j, 10 + 11j, 11 + 12j, 12 + 13j],
        ],
        dtype=np.complex64,
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "clip"})
    )

    # Input 9: Single choice layer, mode='clip'
    a = np.array([0, 5, -3, 2], dtype=np.int32)
    choices = np.array([[10, 20, 30, 40]], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "clip"})
    )

    # Input 10: Broadcasting across dimension with shape 1, mode='wrap'
    a = np.array([[0], [1], [2]], dtype=np.int32)
    choices = np.array(
        [
            [[1, 2, 3, 4]],
            [[5, 6, 7, 8]],
            [[9, 10, 11, 12]],
        ],
        dtype=np.int32,
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "wrap"})
    )

    return list_of_inputs


generated_inputs["jax.numpy.choose_2"] = choose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.choose_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.choose_2'.")


check_valid('jax.numpy.choose', generated_inputs['jax.numpy.choose_2'], lib="jax", suffix=2)
