
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vonmises_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (float32)
    x = np.array(0.5, dtype=np.float32)
    kappa = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "kappa": copy.deepcopy(kappa)})

    # Input 2: 1D arrays of matching shapes
    x = np.linspace(-np.pi, np.pi, 10, dtype=np.float32)
    kappa = np.full((10,), 2.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "kappa": copy.deepcopy(kappa)})

    # Input 3: kappa = 0 (uniform distribution on circle)
    x = np.array([-np.pi / 2, 0.0, np.pi / 2], dtype=np.float64)
    kappa = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "kappa": copy.deepcopy(kappa)})

    # Input 4: 2D arrays (matching shapes)
    x = np.random.uniform(-np.pi, np.pi, size=(4, 5)).astype(np.float32)
    kappa = np.random.uniform(0.1, 5.0, size=(4, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "kappa": copy.deepcopy(kappa)})

    # Input 5: Broadcast 1D x and 2D kappa
    x = np.array([-1.5, 0.0, 1.5], dtype=np.float32)  # shape (3,)
    kappa = np.ones((2, 3), dtype=np.float32)  # shape (2, 3)
    list_of_inputs.append({"x": copy.deepcopy(x), "kappa": copy.deepcopy(kappa)})

    # Input 6: Broadcast 2D x and 1D kappa
    x = np.zeros((3, 4), dtype=np.float64)
    kappa = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64)  # shape (4,)
    list_of_inputs.append({"x": copy.deepcopy(x), "kappa": copy.deepcopy(kappa)})

    # Input 7: 3D arrays
    x = np.random.uniform(-np.pi, np.pi, size=(2, 3, 4)).astype(np.float32)
    kappa = np.random.uniform(0.5, 3.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "kappa": copy.deepcopy(kappa)})

    # Input 8: Scalar x with multi-dimensional kappa
    x = np.array(0.0, dtype=np.float32)
    kappa = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "kappa": copy.deepcopy(kappa)})

    # Input 9: Large kappa values with float64 precision
    x = np.array([-0.01, 0.0, 0.01], dtype=np.float64)
    kappa = np.array([50.0, 100.0, 200.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "kappa": copy.deepcopy(kappa)})

    # Input 10: Column vector x and row vector kappa (outer broadcast)
    x = np.linspace(-np.pi, np.pi, 5, dtype=np.float32).reshape(5, 1)
    kappa = np.array([[0.1, 1.0, 5.0]], dtype=np.float32)  # shape (1, 3)
    list_of_inputs.append({"x": copy.deepcopy(x), "kappa": copy.deepcopy(kappa)})

    # Input 11: 4D tensors
    x = np.zeros((2, 1, 3, 4), dtype=np.float32)
    kappa = np.ones((1, 2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "kappa": copy.deepcopy(kappa)})

    return list_of_inputs


generated_inputs["jax.scipy.stats.vonmises.pdf_1"] = vonmises_pdf_inputs()

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
