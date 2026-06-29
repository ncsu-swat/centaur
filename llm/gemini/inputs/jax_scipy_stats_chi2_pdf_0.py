
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def chi2_pdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays
    list_of_inputs.append({
        "x": np.array([0.5, 1.0, 2.0], dtype=np.float32),
        "df": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    })

    # Input 2: 2D arrays
    list_of_inputs.append({
        "x": np.array([[0.1, 0.5], [1.0, 2.0]], dtype=np.float32),
        "df": np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    })

    # Input 3: float64 1D arrays
    list_of_inputs.append({
        "x": np.array([1.5], dtype=np.float64),
        "df": np.array([5.5], dtype=np.float64),
        "loc": np.array([0.1], dtype=np.float64),
        "scale": np.array([2.0], dtype=np.float64)
    })

    # Input 4: Negative x values
    list_of_inputs.append({
        "x": np.array([-1.0, -0.5, 0.0, 1.0], dtype=np.float32),
        "df": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    })

    # Input 5: Large df
    list_of_inputs.append({
        "x": np.array([50.0, 100.0], dtype=np.float32),
        "df": np.array([100.0, 200.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0], dtype=np.float32)
    })

    # Input 6: 0D (scalar-like) arrays
    list_of_inputs.append({
        "x": np.array(2.5, dtype=np.float32),
        "df": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.5, dtype=np.float32),
        "scale": np.array(1.5, dtype=np.float32)
    })

    # Input 7: 3D random arrays
    list_of_inputs.append({
        "x": np.random.uniform(5.0, 10.0, size=(2, 2, 2)).astype(np.float32),
        "df": np.random.uniform(1.0, 5.0, size=(2, 2, 2)).astype(np.float32),
        "loc": np.random.uniform(0.0, 1.0, size=(2, 2, 2)).astype(np.float32),
        "scale": np.random.uniform(0.5, 2.0, size=(2, 2, 2)).astype(np.float32)
    })

    # Input 8: Broadcasting shapes
    list_of_inputs.append({
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "df": np.array([[2.0], [4.0]], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([1.0], dtype=np.float32)
    })

    # Input 9: Integer-like float arrays with offset loc and scale
    list_of_inputs.append({
        "x": np.array([2.0, 4.0, 6.0], dtype=np.float32),
        "df": np.array([2.0, 4.0, 6.0], dtype=np.float32),
        "loc": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "scale": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    })

    # Input 10: High dimensions (4D)
    list_of_inputs.append({
        "x": np.ones((2, 3, 4, 5), dtype=np.float32),
        "df": np.full((2, 3, 4, 5), 5.0, dtype=np.float32),
        "loc": np.zeros((2, 3, 4, 5), dtype=np.float32),
        "scale": np.ones((2, 3, 4, 5), dtype=np.float32)
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.chi2.pdf"] = chi2_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.chi2.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.chi2.pdf'.")


check_valid('jax.scipy.stats.chi2.pdf', generated_inputs['jax.scipy.stats.chi2.pdf'], lib="jax", suffix=0)
