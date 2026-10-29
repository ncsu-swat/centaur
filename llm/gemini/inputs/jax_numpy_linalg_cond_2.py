
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cond_inputs():
    list_of_inputs = []

    # Input 1: 2D square float32 matrix with p=2.0
    x = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "p": p}))

    # Input 2: 2D square float64 matrix with p=1.0
    x = np.array(
        [[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]], dtype=np.float64
    )
    p = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "p": p}))

    # Input 3: 2D square float32 matrix with p=-1.0
    x = np.array([[3.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
    p = -1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "p": p}))

    # Input 4: 2D square float64 matrix with p=-2.0
    x = np.eye(4, dtype=np.float64) * 2.0 + np.ones((4, 4), dtype=np.float64)
    p = -2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "p": p}))

    # Input 5: 2D square float32 matrix with p=inf
    x = np.array([[10.0, 2.0], [3.0, 8.0]], dtype=np.float32)
    p = float("inf")
    list_of_inputs.append(copy.deepcopy({"x": x, "p": p}))

    # Input 6: 2D square float64 matrix with p=-inf
    x = np.array(
        [[5.0, -2.0, 1.0], [-2.0, 6.0, 0.0], [1.0, 0.0, 4.0]], dtype=np.float64
    )
    p = float("-inf")
    list_of_inputs.append(copy.deepcopy({"x": x, "p": p}))

    # Input 7: 3D batched square float32 matrices with p=2.0
    rng = np.random.RandomState(42)
    x = rng.randn(3, 4, 4).astype(np.float32) + 5.0 * np.eye(
        4, dtype=np.float32
    )
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "p": p}))

    # Input 8: 2D rectangular float32 matrix (M > N) with p=2.0
    x = rng.randn(5, 3).astype(np.float32)
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "p": p}))

    # Input 9: 2D rectangular float64 matrix (M < N) with p=-2.0
    x = rng.randn(3, 5).astype(np.float64)
    p = -2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "p": p}))

    # Input 10: 4D batched square float32 matrices with p=float('inf')
    x = rng.randn(2, 2, 3, 3).astype(np.float32) + 4.0 * np.eye(
        3, dtype=np.float32
    )
    p = float("inf")
    list_of_inputs.append(copy.deepcopy({"x": x, "p": p}))

    # Input 11: 3D batched rectangular float64 matrices with p=2.0
    x = rng.randn(4, 6, 2).astype(np.float64)
    p = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "p": p}))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.cond_2"] = cond_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.cond_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.cond_2'.")


check_valid('jax.numpy.linalg.cond', generated_inputs['jax.numpy.linalg.cond_2'], lib="jax", suffix=2)
