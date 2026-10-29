
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def choose_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D index array with 2D choices, mode='raise'
    a = np.array([0, 1, 2, 0], dtype=np.int32)
    choices = np.array(
        [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]],
        dtype=np.int32,
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "raise"})
    )

    # Input 2: 1D with mode='wrap' and out-of-bound/negative indices
    a = np.array([-1, 0, 1, 2, 3], dtype=np.int32)
    choices = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]],
        dtype=np.float32,
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "wrap"})
    )

    # Input 3: 1D with mode='clip' and out-of-bound indices
    a = np.array([-2, 0, 1, 5], dtype=np.int64)
    choices = np.array(
        [
            [100, 200, 300, 400],
            [500, 600, 700, 800],
            [900, 1000, 1100, 1200],
        ],
        dtype=np.int64,
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "clip"})
    )

    # Input 4: 2D index array with 3D choices, mode='raise'
    a = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int32)
    choices = np.zeros((3, 2, 3), dtype=np.float64)
    choices[0] = 1.0
    choices[1] = 2.0
    choices[2] = 3.0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "raise"})
    )

    # Input 5: 2D index array with mode='wrap'
    a = np.array([[0, 2, -1], [3, 1, 0]], dtype=np.int32)
    choices = np.ones((2, 2, 3), dtype=np.int32)
    choices[1] = 10
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "wrap"})
    )

    # Input 6: 3D index array with 4D choices, mode='clip'
    a = np.random.randint(-3, 5, size=(2, 2, 3)).astype(np.int32)
    choices = np.random.randn(3, 2, 2, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "clip"})
    )

    # Input 7: Boolean choices, mode='raise'
    a = np.array([0, 1, 0, 1], dtype=np.int32)
    choices = np.array(
        [[True, False, True, False], [False, True, False, True]],
        dtype=np.bool_,
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "raise"})
    )

    # Input 8: Complex choices, 2D index array, mode='raise'
    a = np.array([[0, 1], [1, 0]], dtype=np.int32)
    choices = np.array(
        [
            [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
            [[9 + 1j, 2 + 3j], [4 + 5j, 6 + 7j]],
        ],
        dtype=np.complex128,
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "raise"})
    )

    # Input 9: Unsigned integer choices, mode='wrap'
    a = np.array([[0, 5, -2]], dtype=np.int32)
    choices = np.arange(6, dtype=np.uint8).reshape((2, 1, 3))
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "wrap"})
    )

    # Input 10: 1D index array with int16 choices, mode='clip'
    a = np.array([-5, 0, 1, 10], dtype=np.int32)
    choices = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        dtype=np.int16,
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "clip"})
    )

    # Input 11: 3D index array with 4D choices, mode='raise'
    a = np.zeros((2, 1, 4), dtype=np.int32)
    choices = np.ones((2, 2, 1, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "choices": choices, "mode": "raise"})
    )

    return list_of_inputs


generated_inputs["jax.numpy.choose_1"] = choose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.choose_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.choose_1'.")


check_valid('jax.numpy.choose', generated_inputs['jax.numpy.choose_1'], lib="jax", suffix=1)
