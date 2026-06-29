
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def leslie_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays with float32
    f = np.array([0.1, 2.0, 1.0, 0.1], dtype=np.float32)
    s = np.array([0.2, 0.8, 0.7], dtype=np.float32)
    input_dict = {"f": f, "s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Minimal dimension N=2
    f = np.array([1.5, 0.5], dtype=np.float32)
    s = np.array([0.9], dtype=np.float32)
    input_dict = {"f": f, "s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 precision
    f = np.array([0.5, 1.2, 3.0, 2.5, 0.1], dtype=np.float64)
    s = np.array([0.95, 0.85, 0.75, 0.50], dtype=np.float64)
    input_dict = {"f": f, "s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D batch dimension (shape: (3, 4) and (3, 3))
    f = np.random.uniform(0, 3, size=(3, 4)).astype(np.float32)
    s = np.random.uniform(0, 1, size=(3, 3)).astype(np.float32)
    input_dict = {"f": f, "s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D batch dimension (shape: (2, 2, 3) and (2, 2, 2))
    f = np.random.uniform(0, 5, size=(2, 2, 3)).astype(np.float64)
    s = np.random.uniform(0, 1, size=(2, 2, 2)).astype(np.float64)
    input_dict = {"f": f, "s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large size N
    f = np.random.uniform(0, 2, size=(100,)).astype(np.float32)
    s = np.random.uniform(0, 1, size=(99,)).astype(np.float32)
    input_dict = {"f": f, "s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values
    f = np.array([-0.5, -1.0, -2.0], dtype=np.float32)
    s = np.array([-0.1, -0.2], dtype=np.float32)
    input_dict = {"f": f, "s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batch dimension with size 1
    f = np.random.uniform(0, 1, size=(1, 5)).astype(np.float32)
    s = np.random.uniform(0, 1, size=(1, 4)).astype(np.float32)
    input_dict = {"f": f, "s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D batch dimension (shape: (1, 2, 3, 4) and (1, 2, 3, 3))
    f = np.random.uniform(0, 2, size=(1, 2, 3, 4)).astype(np.float32)
    s = np.random.uniform(0, 1, size=(1, 2, 3, 3)).astype(np.float32)
    input_dict = {"f": f, "s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Zero and positive values representing realistic survival/fecundity
    f = np.array([0.0, 0.0, 2.5, 1.8, 0.2], dtype=np.float32)
    s = np.array([0.8, 0.9, 0.7, 0.5], dtype=np.float32)
    input_dict = {"f": f, "s": s}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
