
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def spence_inputs():
    list_of_inputs = []

    # Input 1: Scalar-like (0-D array), float32, positive
    x = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1-D array, float32, positive values
    x = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2-D array, float64, positive values
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3-D array, float32, positive values
    x = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1-D array, float64, positive values
    x = np.array([0.001, 10.0, 100.0], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large 1-D array, float32, uniform distribution
    x = np.random.uniform(1e-5, 1.0, size=(100,)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2-D array, float32, values around 1.0 where spence(1.0) = 0
    x = np.array([[0.99, 1.0, 1.01], [0.95, 1.0, 1.05]], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4-D array, float64, random positive values
    x = np.random.exponential(scale=2.0, size=(2, 2, 2, 2)).astype(np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1-D array, float32, containing extremely small positive values
    x = np.array([1e-10, 1e-8, 1e-6, 1e-4], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1-D array, float64, containing very large values
    x = np.array([1e4, 1e5, 1e6], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.spence"] = spence_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.spence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.spence'.")


check_valid('jax.scipy.special.spence', generated_inputs['jax.scipy.special.spence'], lib="jax", suffix=0)
