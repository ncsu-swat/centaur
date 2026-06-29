
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expm_frechet_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 3x3
    A = np.random.randn(3, 3).astype(np.float32)
    E = np.random.randn(3, 3).astype(np.float32)
    list_of_inputs.append({
        "A": A,
        "E": E,
        "method": "ignored_method",
        "compute_expm": True
    })

    # Input 2: float64 2x2
    A = np.random.randn(2, 2).astype(np.float64)
    E = np.random.randn(2, 2).astype(np.float64)
    list_of_inputs.append({
        "A": A,
        "E": E,
        "method": "Pade",
        "compute_expm": False
    })

    # Input 3: Batched shape (2, 4, 4)
    A = np.random.randn(2, 4, 4).astype(np.float32)
    E = np.random.randn(2, 4, 4).astype(np.float32)
    list_of_inputs.append({
        "A": A,
        "E": E,
        "method": "default",
        "compute_expm": True
    })

    # Input 4: Negative values 5x5
    A = -np.abs(np.random.randn(5, 5)).astype(np.float32)
    E = np.random.randn(5, 5).astype(np.float32)
    list_of_inputs.append({
        "A": A,
        "E": E,
        "method": "",
        "compute_expm": True
    })

    # Input 5: Batched shape (3, 2, 2) float64
    A = np.random.randn(3, 2, 2).astype(np.float64)
    E = np.random.randn(3, 2, 2).astype(np.float64)
    list_of_inputs.append({
        "A": A,
        "E": E,
        "method": "test",
        "compute_expm": False
    })

    # Input 6: 1x1 matrix
    A = np.random.randn(1, 1).astype(np.float32)
    E = np.random.randn(1, 1).astype(np.float32)
    list_of_inputs.append({
        "A": A,
        "E": E,
        "method": "none",
        "compute_expm": True
    })

    # Input 7: Zero and identity matrices
    A = np.zeros((6, 6), dtype=np.float32)
    E = np.eye(6, dtype=np.float32)
    list_of_inputs.append({
        "A": A,
        "E": E,
        "method": "foo",
        "compute_expm": True
    })

    # Input 8: Positive only A, negative only E
    A = np.abs(np.random.randn(4, 4)).astype(np.float64)
    E = -np.abs(np.random.randn(4, 4)).astype(np.float64)
    list_of_inputs.append({
        "A": A,
        "E": E,
        "method": "bar",
        "compute_expm": False
    })

    # Input 9: Batched shape (4, 3, 3)
    A = np.random.randn(4, 3, 3).astype(np.float32)
    E = np.random.randn(4, 3, 3).astype(np.float32)
    list_of_inputs.append({
        "A": A,
        "E": E,
        "method": "multi_batch",
        "compute_expm": True
    })

    # Input 10: 8x8 float32 matrix
    A = np.random.randn(8, 8).astype(np.float32)
    E = np.random.randn(8, 8).astype(np.float32)
    list_of_inputs.append({
        "A": A,
        "E": E,
        "method": "scipy",
        "compute_expm": True
    })

    return list_of_inputs

generated_inputs["jax.scipy.linalg.expm_frechet"] = expm_frechet_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.expm_frechet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.expm_frechet'.")


check_valid('jax.scipy.linalg.expm_frechet', generated_inputs['jax.scipy.linalg.expm_frechet'], lib="jax", suffix=0)
