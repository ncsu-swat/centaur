
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def block_inputs():
    list_of_inputs = []

    # Input 1: 1D list of 1D float32 arrays of identical shape (2,)
    a1 = np.array([1.0, 2.0], dtype=np.float32)
    b1 = np.array([3.0, 4.0], dtype=np.float32)
    c1 = np.array([-1.0, -2.0], dtype=np.float32)
    list_of_inputs.append({"arrays": [a1, b1, c1]})

    # Input 2: 1D list of 1D int32 arrays of shape (3,)
    a2 = np.array([10, -20, 30], dtype=np.int32)
    b2 = np.array([-40, 50, -60], dtype=np.int32)
    list_of_inputs.append({"arrays": [a2, b2]})

    # Input 3: 2D nested list (2x2) of 2D float32 arrays of shape (2, 2)
    zeros = np.zeros((2, 2), dtype=np.float32)
    ones = np.ones((2, 2), dtype=np.float32)
    twos = np.full((2, 2), 2.0, dtype=np.float32)
    threes = np.full((2, 2), -3.0, dtype=np.float32)
    list_of_inputs.append({"arrays": [[zeros, ones], [twos, threes]]})

    # Input 4: 2D nested list (2x2) of 2D float64 arrays of shape (3, 3)
    a4 = np.full((3, 3), 1.5, dtype=np.float64)
    b4 = np.full((3, 3), -2.5, dtype=np.float64)
    c4 = np.full((3, 3), 3.5, dtype=np.float64)
    d4 = np.full((3, 3), -4.5, dtype=np.float64)
    list_of_inputs.append({"arrays": [[a4, b4], [c4, d4]]})

    # Input 5: 2D nested list (2x3) of 2D int64 arrays of shape (2, 2)
    row1 = [
        np.full((2, 2), 1, dtype=np.int64),
        np.full((2, 2), 2, dtype=np.int64),
        np.full((2, 2), 3, dtype=np.int64),
    ]
    row2 = [
        np.full((2, 2), -1, dtype=np.int64),
        np.full((2, 2), -2, dtype=np.int64),
        np.full((2, 2), -3, dtype=np.int64),
    ]
    list_of_inputs.append({"arrays": [row1, row2]})

    # Input 6: 1D list of 2D float32 arrays of shape (4, 4)
    a6 = np.ones((4, 4), dtype=np.float32)
    b6 = np.zeros((4, 4), dtype=np.float32)
    list_of_inputs.append({"arrays": [a6, b6]})

    # Input 7: 3D nested list (2x2x2) of 1D float32 arrays of shape (2,)
    elem = np.array([0.5, -0.5], dtype=np.float32)
    list_of_inputs.append(
        {
            "arrays": [
                [[elem, elem], [elem, elem]],
                [[elem, elem], [elem, elem]],
            ]
        }
    )

    # Input 8: 2D nested list (2x2) of boolean arrays of shape (2, 2)
    b_t = np.ones((2, 2), dtype=bool)
    b_f = np.zeros((2, 2), dtype=bool)
    list_of_inputs.append({"arrays": [[b_t, b_f], [b_f, b_t]]})

    # Input 9: 2D nested list (1x2) of 3D float32 arrays of shape (2, 2, 2)
    blk_a = np.ones((2, 2, 2), dtype=np.float32)
    blk_b = np.zeros((2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"arrays": [[blk_a, blk_b]]})

    # Input 10: 1D list of 1D float16 arrays of shape (1,)
    a10 = np.array([1.5], dtype=np.float16)
    b10 = np.array([-2.5], dtype=np.float16)
    c10 = np.array([0.0], dtype=np.float16)
    list_of_inputs.append({"arrays": [a10, b10, c10]})

    # Input 11: 2D nested list (2x1) of 2D int16 arrays of shape (3, 2)
    a11 = np.full((3, 2), 7, dtype=np.int16)
    b11 = np.full((3, 2), -7, dtype=np.int16)
    list_of_inputs.append({"arrays": [[a11], [b11]]})

    return list_of_inputs


generated_inputs["jax.numpy.block_3"] = block_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.block_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.block_3'.")


check_valid('jax.numpy.block', generated_inputs['jax.numpy.block_3'], lib="jax", suffix=3)
