
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def inv_inputs():
    list_of_inputs = []

    def make_invertible(shape, dtype=np.float32):
        if len(shape) < 2 or shape[-1] != shape[-2]:
            raise ValueError("Must be a square matrix shape")
        
        if np.issubdtype(dtype, np.complexfloating):
            r_dtype = np.float32 if dtype == np.complex64 else np.float64
            r = np.random.randn(*shape).astype(r_dtype)
            i = np.random.randn(*shape).astype(r_dtype)
            a = (r + 1j * i).astype(dtype)
        else:
            a = np.random.randn(*shape).astype(dtype)
            
        n = shape[-1]
        eye = np.eye(n, dtype=dtype)
        batch_shape = shape[:-2]
        for _ in batch_shape:
            eye = np.expand_dims(eye, axis=0)
        eye = np.tile(eye, batch_shape + (1, 1))
        return a + eye * n

    # Input 1: Simple 2x2 float32 matrix
    a = make_invertible((2, 2), np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 2: 3x3 float64 matrix with negative values
    a = make_invertible((3, 3), np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 3: Batched 4x4 float32 matrices (shape: 2, 4, 4)
    a = make_invertible((2, 4, 4), np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 4: Batched 2x2 float64 matrices (shape: 3, 2, 2)
    a = make_invertible((3, 2, 2), np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 5: Higher dimensional batch (shape: 2, 2, 3, 3)
    a = make_invertible((2, 2, 3, 3), np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 6: Large 10x10 float32 matrix
    a = make_invertible((10, 10), np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 7: Trivial 1x1 float32 matrix
    a = make_invertible((1, 1), np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 8: Complex64 3x3 matrix
    a = make_invertible((3, 3), np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 9: Complex128 2x2 matrix
    a = make_invertible((2, 2), np.complex128)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 10: Batched 5x5 complex64 matrices (shape: 1, 5, 5)
    a = make_invertible((1, 5, 5), np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    return list_of_inputs

generated_inputs["jax.numpy.linalg.inv"] = inv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.inv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.inv'.")


check_valid('jax.numpy.linalg.inv', generated_inputs['jax.numpy.linalg.inv'], lib="jax", suffix=0)
