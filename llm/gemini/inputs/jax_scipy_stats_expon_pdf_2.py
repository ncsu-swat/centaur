
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_expon_pdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, standard parameters
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with negative values and shifted loc
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "loc": 0.5, "scale": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, scale > 1
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "loc": 0.0, "scale": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, float64
    x = np.random.uniform(-5, 5, (2, 2, 2)).astype(np.float64)
    input_dict = {"x": x, "loc": -1.0, "scale": 3.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D array (scalar tensor)
    x = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "loc": 0.0, "scale": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array with very small scale
    x = np.array([0.01, 0.02, 0.05], dtype=np.float32)
    input_dict = {"x": x, "loc": 0.0, "scale": 0.01}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large array of size 10, large loc
    x = np.linspace(10, 20, 10, dtype=np.float32)
    input_dict = {"x": x, "loc": 10.0, "scale": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High dimensional 4D array
    x = np.random.exponential(scale=1.0, size=(2, 3, 2, 1)).astype(np.float32)
    input_dict = {"x": x, "loc": 0.2, "scale": 0.8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive values, large scale
    x = np.array([100.0, 200.0, 500.0], dtype=np.float32)
    input_dict = {"x": x, "loc": 50.0, "scale": 100.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with zeros, negative scale offset
    x = np.zeros((3, 3), dtype=np.float32)
    input_dict = {"x": x, "loc": -2.0, "scale": 4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.expon.pdf_2"] = generate_expon_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.pdf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.pdf_2'.")


check_valid('jax.scipy.stats.expon.pdf', generated_inputs['jax.scipy.stats.expon.pdf_2'], lib="jax", suffix=2)
