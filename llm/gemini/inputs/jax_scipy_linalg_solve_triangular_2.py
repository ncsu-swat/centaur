
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def solve_triangular_inputs():
    list_of_inputs = []

    def gen_tri(shape, dtype=np.float32):
        mat = np.random.randn(*shape).astype(dtype)
        n = shape[-1]
        diag_indices = np.diag_indices(n)
        if len(shape) == 2:
            mat[diag_indices] += np.sign(mat[diag_indices]) * 2.0 + 1e-2
        else:
            for i in range(shape[0]):
                mat[i][diag_indices] += np.sign(mat[i][diag_indices]) * 2.0 + 1e-2
        return mat

    # Input 1: Basic upper triangular, 1D RHS
    a = gen_tri((4, 4))
    b = np.random.randn(4).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'trans': 'N', 'lower': False, 
        'unit_diagonal': False, 'overwrite_b': False, 'debug': False, 'check_finite': True
    })

    # Input 2: Lower triangular, 1D RHS
    a = gen_tri((5, 5))
    b = np.random.randn(5).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'trans': 'N', 'lower': True, 
        'unit_diagonal': False, 'overwrite_b': False, 'debug': False, 'check_finite': True
    })

    # Input 3: Transposed, upper triangular
    a = gen_tri((3, 3))
    b = np.random.randn(3).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'trans': 'T', 'lower': False, 
        'unit_diagonal': False, 'overwrite_b': True, 'debug': False, 'check_finite': True
    })

    # Input 4: Conjugate transpose (using complex numbers)
    a = (gen_tri((4, 4)) + 1j * gen_tri((4, 4))).astype(np.complex128)
    b = (np.random.randn(4, 2) + 1j * np.random.randn(4, 2)).astype(np.complex128)
    list_of_inputs.append({
        'a': a, 'b': b, 'trans': 'C', 'lower': True, 
        'unit_diagonal': False, 'overwrite_b': False, 'debug': False, 'check_finite': True
    })

    # Input 5: Batched matrices
    a = gen_tri((2, 3, 3))
    b = np.random.randn(2, 3, 2).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'trans': 'N', 'lower': False, 
        'unit_diagonal': False, 'overwrite_b': False, 'debug': True, 'check_finite': False
    })

    # Input 6: Unit diagonal, lower triangular
    a = gen_tri((6, 6))
    b = np.random.randn(6).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'trans': 'N', 'lower': True, 
        'unit_diagonal': True, 'overwrite_b': False, 'debug': False, 'check_finite': True
    })

    # Input 7: Float64 type, negative values
    a = -np.abs(gen_tri((5, 5), dtype=np.float64))
    b = np.random.randn(5, 3).astype(np.float64)
    list_of_inputs.append({
        'a': a, 'b': b, 'trans': 'T', 'lower': True, 
        'unit_diagonal': False, 'overwrite_b': False, 'debug': False, 'check_finite': True
    })

    # Input 8: Larger size, upper triangular
    a = gen_tri((32, 32))
    b = np.random.randn(32, 5).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'trans': 'N', 'lower': False, 
        'unit_diagonal': False, 'overwrite_b': False, 'debug': False, 'check_finite': True
    })

    # Input 9: Unit diagonal, 1D RHS
    a = gen_tri((10, 10))
    b = np.random.randn(10).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'trans': 'N', 'lower': True, 
        'unit_diagonal': True, 'overwrite_b': False, 'debug': False, 'check_finite': True
    })

    # Input 10: Batched with trans='C', float32
    a = gen_tri((3, 4, 4))
    b = np.random.randn(3, 4, 1).astype(np.float32)
    list_of_inputs.append({
        'a': a, 'b': b, 'trans': 'C', 'lower': False, 
        'unit_diagonal': False, 'overwrite_b': False, 'debug': False, 'check_finite': True
    })

    return list_of_inputs

generated_inputs["jax.scipy.linalg.solve_triangular_2"] = solve_triangular_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.solve_triangular_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.solve_triangular_2'.")


check_valid('jax.scipy.linalg.solve_triangular', generated_inputs['jax.scipy.linalg.solve_triangular_2'], lib="jax", suffix=2)
