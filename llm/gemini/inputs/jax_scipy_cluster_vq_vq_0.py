
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vq_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 arrays
    obs = np.array([[1.1, 2.1, 3.1], [5.9, 4.8, 6.2]], dtype=np.float32)
    code_book = np.array(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
        dtype=np.float32,
    )
    list_of_inputs.append(
        copy.deepcopy({"obs": obs, "code_book": code_book, "check_finite": True})
    )

    # Input 2: 1D arrays for both obs and code_book
    obs = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    code_book = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"obs": obs, "code_book": code_book, "check_finite": False}
        )
    )

    # Input 3: Float64 inputs with negative values
    obs = np.array([[-1.5, -2.5], [0.0, 1.0], [3.2, -4.1]], dtype=np.float64)
    code_book = np.array(
        [[-1.0, -2.0], [0.5, 0.5], [3.0, -4.0]], dtype=np.float64
    )
    list_of_inputs.append(
        copy.deepcopy({"obs": obs, "code_book": code_book, "check_finite": True})
    )

    # Input 4: 2D arrays with single feature column
    obs = np.array([[-2.0], [0.0], [2.0]], dtype=np.float32)
    code_book = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"obs": obs, "code_book": code_book, "check_finite": False}
        )
    )

    # Input 5: 1D arrays with negative values
    obs = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    code_book = np.array([-8.0, 0.0, 8.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"obs": obs, "code_book": code_book, "check_finite": True})
    )

    # Input 6: Single observation and single codebook entry (both 2D)
    obs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    code_book = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"obs": obs, "code_book": code_book, "check_finite": False}
        )
    )

    # Input 7: Higher-dimensional features (large N)
    obs = np.random.randn(20, 64).astype(np.float32)
    code_book = np.random.randn(8, 64).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"obs": obs, "code_book": code_book, "check_finite": True})
    )

    # Input 8: Integer inputs
    obs = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    code_book = np.array([[0, 0], [2, 3], [5, 5]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"obs": obs, "code_book": code_book, "check_finite": True}
        )
    )

    # Input 9: Large number of observations, small number of codes
    obs = np.random.uniform(-10.0, 10.0, size=(100, 4)).astype(np.float64)
    code_book = np.random.uniform(-10.0, 10.0, size=(5, 4)).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"obs": obs, "code_book": code_book, "check_finite": False}
        )
    )

    # Input 10: Single 1D elements
    obs = np.array([4.2], dtype=np.float32)
    code_book = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"obs": obs, "code_book": code_book, "check_finite": True})
    )

    # Input 11: Zeros array test
    obs = np.zeros((5, 3), dtype=np.float32)
    code_book = np.ones((2, 3), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"obs": obs, "code_book": code_book, "check_finite": False}
        )
    )

    return list_of_inputs


generated_inputs["jax.scipy.cluster.vq.vq"] = vq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.cluster.vq.vq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.cluster.vq.vq'.")


check_valid('jax.scipy.cluster.vq.vq', generated_inputs['jax.scipy.cluster.vq.vq'], lib="jax", suffix=0)
