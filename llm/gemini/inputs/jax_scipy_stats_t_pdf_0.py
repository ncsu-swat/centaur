
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_t_pdf_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D arrays, float32
    list_of_inputs.append({
        "x": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "df": np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    })

    # Input 2: 0D arrays (scalars represented as numpy arrays), float32
    list_of_inputs.append({
        "x": np.array(1.5, dtype=np.float32),
        "df": np.array(2.5, dtype=np.float32),
        "loc": np.array(-0.5, dtype=np.float32),
        "scale": np.array(2.0, dtype=np.float32)
    })

    # Input 3: 2D arrays with random values, float64
    list_of_inputs.append({
        "x": np.random.randn(3, 4).astype(np.float64),
        "df": np.abs(np.random.randn(3, 4)).astype(np.float64) + 0.5,
        "loc": np.random.randn(3, 4).astype(np.float64),
        "scale": np.abs(np.random.randn(3, 4)).astype(np.float64) + 0.1
    })

    # Input 4: Broadcasting shapes: x is 2D, df is 1D, loc/scale are 1D
    list_of_inputs.append({
        "x": np.random.randn(2, 3).astype(np.float32),
        "df": np.array([1.0, 2.0, 5.0], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([1.5], dtype=np.float32)
    })

    # Input 5: Large degrees of freedom (t-dist approaching normal-like)
    list_of_inputs.append({
        "x": np.array([-10.0, 0.0, 10.0], dtype=np.float32),
        "df": np.array([100.0, 100.0, 100.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    })

    # Input 6: Very small degrees of freedom (heavy-tailed)
    list_of_inputs.append({
        "x": np.array([-100.0, 0.0, 100.0], dtype=np.float32),
        "df": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    })

    # Input 7: 3D arrays, float32
    list_of_inputs.append({
        "x": np.random.randn(2, 2, 2).astype(np.float32),
        "df": np.ones((2, 2, 2), dtype=np.float32) * 4.0,
        "loc": np.zeros((2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2), dtype=np.float32) * 0.5
    })

    # Input 8: Large scale parameter
    list_of_inputs.append({
        "x": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "df": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([10.0, 10.0, 10.0], dtype=np.float32)
    })

    # Input 9: Small scale parameter
    list_of_inputs.append({
        "x": np.array([0.01, 0.02, 0.03], dtype=np.float32),
        "df": np.array([1.5, 1.5, 1.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([0.01, 0.01, 0.01], dtype=np.float32)
    })

    # Input 10: Highly asymmetric dimensions with broadcasting, float64
    list_of_inputs.append({
        "x": np.array([[1.0], [2.0]], dtype=np.float64),
        "df": np.array([3.0, 4.0], dtype=np.float64),
        "loc": np.array([[-1.0, 1.0]], dtype=np.float64),
        "scale": np.array([[0.5, 2.0]], dtype=np.float64)
    })

    return list_of_inputs

generated_inputs["jax.scipy.stats.t.pdf"] = jax_scipy_stats_t_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.t.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.t.pdf'.")


check_valid('jax.scipy.stats.t.pdf', generated_inputs['jax.scipy.stats.t.pdf'], lib="jax", suffix=0)
