
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cholesky_inputs():
    list_of_inputs = []

    def make_spd(shape, dtype=np.float32):
        N = shape[-1]
        B = np.random.randn(*shape).astype(dtype)
        A = np.matmul(B, np.swapaxes(B, -1, -2))
        eye = np.eye(N, dtype=dtype)
        batch_shape = shape[:-2]
        for _ in range(len(batch_shape)):
            eye = np.expand_dims(eye, axis=0)
        eye = np.tile(eye, batch_shape + (1, 1))
        return A + (N * 2.0 + 1.0) * eye

    # Input 1: 2D small matrix, default options
    a = make_spd((3, 3), np.float32)
    list_of_inputs.append({
        "a": a,
        "upper": False,
        "symmetrize_input": True
    })

    # Input 2: 2D small matrix, upper=True
    a = make_spd((4, 4), np.float32)
    list_of_inputs.append({
        "a": a,
        "upper": True,
        "symmetrize_input": True
    })

    # Input 3: float64 matrix
    a = make_spd((5, 5), np.float64)
    list_of_inputs.append({
        "a": a,
        "upper": False,
        "symmetrize_input": True
    })

    # Input 4: 3D batched matrix, symmetrize_input=False
    a = make_spd((2, 3, 3), np.float32)
    list_of_inputs.append({
        "a": a,
        "upper": False,
        "symmetrize_input": False
    })

    # Input 5: 3D batched, upper=True
    a = make_spd((3, 4, 4), np.float64)
    list_of_inputs.append({
        "a": a,
        "upper": True,
        "symmetrize_input": True
    })

    # Input 6: 4D batched matrix
    a = make_spd((2, 2, 3, 3), np.float32)
    list_of_inputs.append({
        "a": a,
        "upper": True,
        "symmetrize_input": False
    })

    # Input 7: 1x1 trivial matrix
    a = make_spd((1, 1), np.float32)
    list_of_inputs.append({
        "a": a,
        "upper": False,
        "symmetrize_input": True
    })

    # Input 8: larger 2D matrix
    a = make_spd((20, 20), np.float32)
    list_of_inputs.append({
        "a": a,
        "upper": False,
        "symmetrize_input": True
    })

    # Input 9: float64, 4D batched
    a = make_spd((1, 3, 2, 2), np.float64)
    list_of_inputs.append({
        "a": a,
        "upper": True,
        "symmetrize_input": True
    })

    # Input 10: 10x10 matrix, symmetrize_input=False
    a = make_spd((10, 10), np.float64)
    list_of_inputs.append({
        "a": a,
        "upper": False,
        "symmetrize_input": False
    })

    return list_of_inputs

generated_inputs["jax.numpy.linalg.cholesky"] = cholesky_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.cholesky' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.cholesky'.")


check_valid('jax.numpy.linalg.cholesky', generated_inputs['jax.numpy.linalg.cholesky'], lib="jax", suffix=0)
