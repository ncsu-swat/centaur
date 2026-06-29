
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gennorm_pdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, positive beta
    x = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    beta = np.array([1.5, 1.5, 1.5], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D arrays (scalars as tensors)
    x = np.array(0.5, dtype=np.float32)
    beta = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float32
    x = np.random.randn(3, 3).astype(np.float32)
    beta = np.ones((3, 3), dtype=np.float32) * 2.5
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D arrays, float32
    x = np.random.randn(2, 2, 2).astype(np.float32)
    beta = np.full((2, 2, 2), 3.0, dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 precision
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    beta = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Small beta value (beta > 0)
    x = np.array([-0.5, 0.5], dtype=np.float32)
    beta = np.array([0.1, 0.1], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large beta value
    x = np.array([-1.0, 1.0], dtype=np.float32)
    beta = np.array([20.0, 20.0], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting x (1D) and beta (2D)
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    beta = np.array([[1.5], [2.5]], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive and negative values for x
    x = np.array([-10.0, 10.0, -100.0, 100.0], dtype=np.float32)
    beta = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed positive beta values in beta array
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    beta = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "beta": beta}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gennorm.pdf"] = gennorm_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gennorm.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gennorm.pdf'.")


check_valid('jax.scipy.stats.gennorm.pdf', generated_inputs['jax.scipy.stats.gennorm.pdf'], lib="jax", suffix=0)
