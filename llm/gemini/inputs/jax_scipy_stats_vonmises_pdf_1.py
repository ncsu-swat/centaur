
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_vonmises_pdf_inputs():
    list_of_inputs = []

    # Input 1: Scalar/0D arrays, float32
    x = np.array(0.5, dtype=np.float32)
    kappa = np.array(1.0, dtype=np.float32)
    input_dict = {"x": x, "kappa": kappa}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays, float32, standard range
    x = np.array([-np.pi, -1.0, 0.0, 1.0, np.pi], dtype=np.float32)
    kappa = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    input_dict = {"x": x, "kappa": kappa}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float64, random values
    x = np.random.uniform(-np.pi, np.pi, (3, 3)).astype(np.float64)
    kappa = np.random.uniform(0.1, 5.0, (3, 3)).astype(np.float64)
    input_dict = {"x": x, "kappa": kappa}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D arrays, float32, values outside typical [-pi, pi] interval (periodic behavior)
    x = np.random.uniform(-2 * np.pi, 2 * np.pi, (2, 2, 2)).astype(np.float32)
    kappa = np.random.uniform(0.0, 10.0, (2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "kappa": kappa}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting arrays (3, 1) and (1, 3)
    x = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
    kappa = np.array([[0.1, 1.0, 10.0]], dtype=np.float32)
    input_dict = {"x": x, "kappa": kappa}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting 2D x with 1D kappa
    x = np.random.uniform(-np.pi, np.pi, (4, 4)).astype(np.float32)
    kappa = np.array([2.0], dtype=np.float32)
    input_dict = {"x": x, "kappa": kappa}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Uniform case where kappa is 0
    x = np.linspace(-np.pi, np.pi, 10).astype(np.float32)
    kappa = np.zeros(10, dtype=np.float32)
    input_dict = {"x": x, "kappa": kappa}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64 with large/small values
    x = np.array([10.0, -10.0], dtype=np.float64)
    kappa = np.array([0.01, 0.01], dtype=np.float64)
    input_dict = {"x": x, "kappa": kappa}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High dimension 4D arrays
    x = np.random.uniform(-1.0, 1.0, (2, 2, 2, 2)).astype(np.float32)
    kappa = np.random.uniform(1.0, 2.0, (2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "kappa": kappa}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High concentration (large kappa)
    x = np.array([0.01, -0.01, 0.0], dtype=np.float32)
    kappa = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    input_dict = {"x": x, "kappa": kappa}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.vonmises.pdf_1"] = jax_scipy_stats_vonmises_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.vonmises.pdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.vonmises.pdf_1'.")


check_valid('jax.scipy.stats.vonmises.pdf', generated_inputs['jax.scipy.stats.vonmises.pdf_1'], lib="jax", suffix=1)
