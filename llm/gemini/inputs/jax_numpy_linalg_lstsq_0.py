
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lstsq_inputs():
    list_of_inputs = []

    # Input 1: Overdetermined system with 1D right-hand side (float32)
    a = np.array([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]], dtype=np.float32)
    b = np.array([1.0, 2.0, -3.0], dtype=np.float32)
    input_dict = {"a": a, "b": b, "rcond": 1e-5, "numpy_resid": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Overdetermined system with 2D right-hand side (float32)
    a = np.random.randn(5, 3).astype(np.float32)
    b = np.random.randn(5, 2).astype(np.float32)
    input_dict = {"a": a, "b": b, "rcond": 1e-6, "numpy_resid": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Underdetermined system with 1D right-hand side (float64)
    a = np.random.randn(3, 6).astype(np.float64)
    b = np.random.randn(3).astype(np.float64)
    input_dict = {"a": a, "b": b, "rcond": 1e-10, "numpy_resid": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Square system with 1D right-hand side (float32)
    a = np.random.randn(4, 4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    input_dict = {"a": a, "b": b, "rcond": 0.0, "numpy_resid": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Square system with multiple right-hand sides (float64)
    a = np.random.randn(6, 6).astype(np.float64)
    b = np.random.randn(6, 4).astype(np.float64)
    input_dict = {"a": a, "b": b, "rcond": 1e-12, "numpy_resid": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex-valued system with 1D right-hand side (complex64)
    a = (np.random.randn(6, 3) + 1j * np.random.randn(6, 3)).astype(np.complex64)
    b = (np.random.randn(6) + 1j * np.random.randn(6)).astype(np.complex64)
    input_dict = {"a": a, "b": b, "rcond": 1e-4, "numpy_resid": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex-valued system with 2D right-hand side (complex128)
    a = (np.random.randn(5, 2) + 1j * np.random.randn(5, 2)).astype(np.complex128)
    b = (np.random.randn(5, 3) + 1j * np.random.randn(5, 3)).astype(np.complex128)
    input_dict = {"a": a, "b": b, "rcond": 1e-8, "numpy_resid": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Rank-deficient matrix with negative values (float64)
    base = np.random.randn(6, 2).astype(np.float64)
    a = np.column_stack([base, base[:, :1] * -2.0, np.random.randn(6, 1).astype(np.float64)])
    b = np.random.randn(6).astype(np.float64)
    input_dict = {"a": a, "b": b, "rcond": 1e-3, "numpy_resid": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small 2x1 matrix (float32)
    a = np.array([[-3.5], [7.2]], dtype=np.float32)
    b = np.array([1.5, -4.0], dtype=np.float32)
    input_dict = {"a": a, "b": b, "rcond": 0.01, "numpy_resid": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger overdetermined system (float32)
    a = np.random.randn(20, 5).astype(np.float32)
    b = np.random.randn(20, 8).astype(np.float32)
    input_dict = {"a": a, "b": b, "rcond": 1e-7, "numpy_resid": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.lstsq"] = lstsq_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.lstsq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.lstsq'.")


check_valid('jax.numpy.linalg.lstsq', generated_inputs['jax.numpy.linalg.lstsq'], lib="jax", suffix=0)
