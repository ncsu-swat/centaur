
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def chi2_pdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays with float32
    x = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    df = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "df": df, "loc": loc, "scale": scale})
    )

    # Input 2: 0D scalars (tensors with shape ())
    x = np.array(3.5, dtype=np.float64)
    df = np.array(4.0, dtype=np.float64)
    loc = np.array(0.0, dtype=np.float64)
    scale = np.array(1.5, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "df": df, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays with broadcasting
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    df = np.array([[3.0], [5.0]], dtype=np.float32)
    loc = np.array([[0.0]], dtype=np.float32)
    scale = np.array([[1.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "df": df, "loc": loc, "scale": scale})
    )

    # Input 4: Negative x values (outside support, pdf returns 0)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    df = np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "df": df, "loc": loc, "scale": scale})
    )

    # Input 5: Non-zero location and scale parameters
    x = np.array([2.5, 4.0, 6.5], dtype=np.float64)
    df = np.array([2.0, 4.0, 6.0], dtype=np.float64)
    loc = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    scale = np.array([0.5, 1.5, 2.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "df": df, "loc": loc, "scale": scale})
    )

    # Input 6: 3D tensors
    x = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    df = np.ones((2, 3, 4), dtype=np.float32) * 5.0
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "df": df, "loc": loc, "scale": scale})
    )

    # Input 7: Fractional degrees of freedom
    x = np.array([0.1, 0.5, 1.2, 3.4], dtype=np.float32)
    df = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "df": df, "loc": loc, "scale": scale})
    )

    # Input 8: Integer-typed arrays (promoted internally)
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    df = np.array([2, 2, 2, 2], dtype=np.int32)
    loc = np.array([0, 0, 0, 0], dtype=np.int32)
    scale = np.array([1, 1, 1, 1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "df": df, "loc": loc, "scale": scale})
    )

    # Input 9: Large degrees of freedom and float64 precision
    x = np.array([90.0, 100.0, 110.0], dtype=np.float64)
    df = np.array([100.0, 100.0, 100.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "df": df, "loc": loc, "scale": scale})
    )

    # Input 10: Multi-dimensional broadcasting across dimensions
    x = np.linspace(0.1, 10.0, 12, dtype=np.float32).reshape(3, 4)
    df = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 4)
    loc = np.array([0.0, 0.5, 1.0], dtype=np.float32).reshape(3, 1)
    scale = np.array([1.0], dtype=np.float32).reshape(1, 1)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "df": df, "loc": loc, "scale": scale})
    )

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
