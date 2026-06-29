
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_scipy_stats_t_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, standard values
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    df = np.array([1.5, 2.0, 3.0, 10.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 2: 2D arrays, float64
    x = np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float64)
    df = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float64)
    loc = np.array([[-0.5, 0.5], [-0.5, 0.5]], dtype=np.float64)
    scale = np.array([[1.5, 1.5], [1.5, 1.5]], dtype=np.float64)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 3: 0D arrays (scalar arrays)
    x = np.array(0.5, dtype=np.float32)
    df = np.array(2.5, dtype=np.float32)
    loc = np.array(-1.0, dtype=np.float32)
    scale = np.array(0.8, dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 4: Broadcasting (x shape (3, 1), loc shape (1, 2))
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    df = np.array([2.0], dtype=np.float32)
    loc = np.array([[0.0, 1.0]], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 5: High dimensions (3D)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    df = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    loc = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 6: Very large degrees of freedom (t -> normal)
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    df = np.array([1000.0, 10000.0, 100000.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 7: Small degrees of freedom (df < 1)
    x = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    df = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 8: High dimensional broadcasting
    x = np.random.randn(1, 5).astype(np.float64)
    df = np.array([[3.0]], dtype=np.float64)
    loc = np.random.randn(3, 1).astype(np.float64)
    scale = np.random.uniform(0.5, 2.5, size=(3, 5)).astype(np.float64)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 9: Large scales
    x = np.array([100.0, -100.0], dtype=np.float32)
    df = np.array([4.0, 4.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([50.0, 50.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    # Input 10: Integer-like input values as float tensors
    x = np.array([-3, 0, 3], dtype=np.float32)
    df = np.array([5, 5, 5], dtype=np.float32)
    loc = np.array([-1, 0, 1], dtype=np.float32)
    scale = np.array([2, 2, 2], dtype=np.float32)
    list_of_inputs.append({"x": x, "df": df, "loc": loc, "scale": scale})

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.scipy.stats.t.logpdf"] = jax_scipy_stats_t_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.t.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.t.logpdf'.")


check_valid('jax.scipy.stats.t.logpdf', generated_inputs['jax.scipy.stats.t.logpdf'], lib="jax", suffix=0)
