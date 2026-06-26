
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def owens_t_inputs():
    list_of_inputs = []

    # Input 1: Scalar-like 0D arrays, float32
    h = np.array(0.5, dtype=np.float32)
    a = np.array(1.5, dtype=np.float32)
    input_dict = {"h": h, "a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, positive values, float32
    h = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    a = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    input_dict = {"h": h, "a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays, negative values, float64
    h = np.array([-1.0, -2.5, -0.5], dtype=np.float64)
    a = np.array([-2.0, -1.0, -0.1], dtype=np.float64)
    input_dict = {"h": h, "a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, mixed positive/negative/zero, float32
    h = np.array([[0.0, -1.2], [2.3, -0.5]], dtype=np.float32)
    a = np.array([[1.0, 2.0], [-1.0, 0.0]], dtype=np.float32)
    input_dict = {"h": h, "a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcastable shapes, 2D and 1D
    h = np.array([[0.1], [0.5], [1.0]], dtype=np.float32) 
    a = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32) 
    input_dict = {"h": h, "a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger dimensions (3D arrays), float64
    h = np.random.uniform(-3.0, 3.0, (2, 3, 4)).astype(np.float64)
    a = np.random.uniform(-3.0, 3.0, (2, 3, 4)).astype(np.float64)
    input_dict = {"h": h, "a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small values close to 0
    h = np.array([1e-5, -1e-5], dtype=np.float32)
    a = np.array([1e-5, -1e-5], dtype=np.float32)
    input_dict = {"h": h, "a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large values for h
    h = np.array([10.0, 20.0, 50.0], dtype=np.float32)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"h": h, "a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values for a, float64
    h = np.array([0.5, 1.0], dtype=np.float64)
    a = np.array([100.0, 500.0], dtype=np.float64)
    input_dict = {"h": h, "a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Random uniform distribution, 1D
    h = np.random.randn(10).astype(np.float32)
    a = np.random.randn(10).astype(np.float32)
    input_dict = {"h": h, "a": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.owens_t"] = owens_t_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.owens_t' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.owens_t'.")


check_valid('jax.scipy.special.owens_t', generated_inputs['jax.scipy.special.owens_t'], lib="jax", suffix=0)
