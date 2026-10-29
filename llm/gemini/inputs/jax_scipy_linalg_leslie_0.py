
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def leslie_inputs():
    list_of_inputs = []

    # Input 1: Minimal size N=2, 1D float32
    f = np.array([1.5, 0.5], dtype=np.float32)
    s = np.array([0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"f": f, "s": s}))

    # Input 2: Typical 1D float32 from documentation (N=4)
    f = np.array([0.1, 2.0, 1.0, 0.1], dtype=np.float32)
    s = np.array([0.2, 0.8, 0.7], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"f": f, "s": s}))

    # Input 3: 1D float64 with negative values (N=5)
    f = np.array([-0.5, 1.2, -0.3, 2.1, 0.0], dtype=np.float64)
    s = np.array([0.4, -0.6, 0.9, 0.1], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"f": f, "s": s}))

    # Input 4: 1D integer values (N=3)
    f = np.array([2, 5, 1], dtype=np.int32)
    s = np.array([1, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"f": f, "s": s}))

    # Input 5: 1D int64 (N=6)
    f = np.array([0, 10, 20, 15, 5, 0], dtype=np.int64)
    s = np.array([1, 2, 3, 2, 1], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"f": f, "s": s}))

    # Input 6: 2D batched inputs (2, 3) and (2, 2) float32
    f = np.random.uniform(0, 5, size=(2, 3)).astype(np.float32)
    s = np.random.uniform(0, 1, size=(2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"f": f, "s": s}))

    # Input 7: 2D batched inputs (3, 5) and (3, 4) float64
    f = np.random.randn(3, 5).astype(np.float64)
    s = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"f": f, "s": s}))

    # Input 8: 3D batched inputs (2, 2, 4) and (2, 2, 3) float32
    f = np.random.uniform(0, 2, size=(2, 2, 4)).astype(np.float32)
    s = np.random.uniform(0, 1, size=(2, 2, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"f": f, "s": s}))

    # Input 9: Complex numbers (complex64, N=3)
    f = np.array([1.0 + 2.0j, 0.5 - 1.0j, 3.0 + 0.0j], dtype=np.complex64)
    s = np.array([0.5 + 0.5j, 0.8 - 0.2j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"f": f, "s": s}))

    # Input 10: Complex numbers (complex128, N=4)
    f = np.array([0.2 - 0.1j, 1.5 + 0.3j, -0.5j, 2.0], dtype=np.complex128)
    s = np.array([0.9, 0.6 + 0.1j, -0.4j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"f": f, "s": s}))

    # Input 11: Larger size N=8, 1D float32
    f = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    s = np.linspace(0.1, 0.9, 7, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"f": f, "s": s}))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.leslie"] = leslie_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.leslie' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.leslie'.")


check_valid('jax.scipy.linalg.leslie', generated_inputs['jax.scipy.linalg.leslie'], lib="jax", suffix=0)
