
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def wrapcauchy_pdf_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D arrays
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    c = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D arrays (scalars)
    x = np.array(1.5, dtype=np.float32)
    c = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array for x, 0D array for c
    x = np.random.uniform(0, 2 * np.pi, size=(3, 3)).astype(np.float32)
    c = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64 1D arrays
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    c = np.array([0.2, 0.4, 0.6], dtype=np.float64)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays with random values
    x = np.random.uniform(-np.pi, np.pi, size=(2, 3, 4)).astype(np.float32)
    c = np.random.uniform(0.1, 0.9, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcastable shapes (3, 1) and (1, 4)
    x = np.random.uniform(0, np.pi, size=(3, 1)).astype(np.float32)
    c = np.random.uniform(0.1, 0.9, size=(1, 4)).astype(np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values in x
    x = np.array([-2.0, -1.0, -0.5], dtype=np.float32)
    c = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large values in x (greater than 2*pi)
    x = np.array([10.0, 20.0, 100.0], dtype=np.float32)
    c = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High dimension 4D arrays
    x = np.random.uniform(0, 2 * np.pi, size=(2, 2, 2, 2)).astype(np.float32)
    c = np.random.uniform(0.01, 0.99, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Values of c close to boundary limits (0 and 1)
    x = np.array([0.0, np.pi, 2 * np.pi], dtype=np.float32)
    c = np.array([0.01, 0.99, 0.5], dtype=np.float32)
    input_dict = {"x": x, "c": c}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.wrapcauchy.pdf"] = wrapcauchy_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.wrapcauchy.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.wrapcauchy.pdf'.")


check_valid('jax.scipy.stats.wrapcauchy.pdf', generated_inputs['jax.scipy.stats.wrapcauchy.pdf'], lib="jax", suffix=0)
