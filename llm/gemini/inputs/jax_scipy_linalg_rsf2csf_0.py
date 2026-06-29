
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import scipy.linalg
import copy

def rsf2csf_inputs():
    list_of_inputs = []

    def generate_schur_pair(shape, dtype=np.float32):
        if len(shape) == 2:
            N = shape[0]
            A = np.random.randn(N, N).astype(dtype)
            T, Z = scipy.linalg.schur(A, output='real')
            return T.astype(dtype), Z.astype(dtype)
        else:
            batch_dims = shape[:-2]
            N = shape[-1]
            flat_size = int(np.prod(batch_dims))
            T_list = []
            Z_list = []
            for _ in range(flat_size):
                A = np.random.randn(N, N).astype(dtype)
                T, Z = scipy.linalg.schur(A, output='real')
                T_list.append(T)
                Z_list.append(Z)
            T_out = np.array(T_list).reshape(shape).astype(dtype)
            Z_out = np.array(Z_list).reshape(shape).astype(dtype)
            return T_out, Z_out

    # Input 1: 3x3, float32, check_finite=True
    T, Z = generate_schur_pair((3, 3), np.float32)
    list_of_inputs.append({"T": T, "Z": Z, "check_finite": True})

    # Input 2: 2x2, float64, check_finite=True
    T, Z = generate_schur_pair((2, 2), np.float64)
    list_of_inputs.append({"T": T, "Z": Z, "check_finite": True})

    # Input 3: 4x4, float32, check_finite=False
    T, Z = generate_schur_pair((4, 4), np.float32)
    list_of_inputs.append({"T": T, "Z": Z, "check_finite": False})

    # Input 4: 5x5, float64, check_finite=False
    T, Z = generate_schur_pair((5, 5), np.float64)
    list_of_inputs.append({"T": T, "Z": Z, "check_finite": False})

    # Input 5: 1x1, float32, check_finite=True
    T, Z = generate_schur_pair((1, 1), np.float32)
    list_of_inputs.append({"T": T, "Z": Z, "check_finite": True})

    # Input 6: Batch (2, 3, 3), float32, check_finite=True
    T, Z = generate_schur_pair((2, 3, 3), np.float32)
    list_of_inputs.append({"T": T, "Z": Z, "check_finite": True})

    # Input 7: Batch (4, 2, 2), float64, check_finite=False
    T, Z = generate_schur_pair((4, 2, 2), np.float64)
    list_of_inputs.append({"T": T, "Z": Z, "check_finite": False})

    # Input 8: Batch (2, 2, 4, 4), float32, check_finite=True
    T, Z = generate_schur_pair((2, 2, 4, 4), np.float32)
    list_of_inputs.append({"T": T, "Z": Z, "check_finite": True})

    # Input 9: 10x10, float32, check_finite=True
    T, Z = generate_schur_pair((10, 10), np.float32)
    list_of_inputs.append({"T": T, "Z": Z, "check_finite": True})

    # Input 10: 8x8, float64, check_finite=False
    T, Z = generate_schur_pair((8, 8), np.float64)
    list_of_inputs.append({"T": T, "Z": Z, "check_finite": False})

    return list_of_inputs

generated_inputs["jax.scipy.linalg.rsf2csf"] = rsf2csf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.rsf2csf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.rsf2csf'.")


check_valid('jax.scipy.linalg.rsf2csf', generated_inputs['jax.scipy.linalg.rsf2csf'], lib="jax", suffix=0)
