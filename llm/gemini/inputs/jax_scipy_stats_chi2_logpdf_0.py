
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def chi2_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar arrays
    list_of_inputs.append(
        {
            "x": np.array(2.5, dtype=np.float32),
            "df": np.array(3.0, dtype=np.float32),
            "loc": np.array(0.0, dtype=np.float32),
            "scale": np.array(1.0, dtype=np.float32),
        }
    )

    # Input 2: 1-D arrays, float64
    list_of_inputs.append(
        {
            "x": np.array([1.0, 2.0, 5.0, 10.0], dtype=np.float64),
            "df": np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64),
            "loc": np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float64),
            "scale": np.array([1.0, 1.5, 2.0, 0.5], dtype=np.float64),
        }
    )

    # Input 3: 2-D arrays with same shape
    list_of_inputs.append(
        {
            "x": np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
            "df": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "loc": np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
            "scale": np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
        }
    )

    # Input 4: Broadcasting shapes (1-D and 2-D)
    list_of_inputs.append(
        {
            "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            "df": np.array([2.0, 3.0, 4.0], dtype=np.float32),
            "loc": np.array([[0.0], [1.0]], dtype=np.float32),
            "scale": np.array([1.0], dtype=np.float32),
        }
    )

    # Input 5: x with values below loc (testing support boundary)
    list_of_inputs.append(
        {
            "x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
            "df": np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32),
            "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        }
    )

    # Input 6: 3-D arrays
    list_of_inputs.append(
        {
            "x": np.ones((2, 3, 4), dtype=np.float32) * 3.0,
            "df": np.ones((2, 3, 4), dtype=np.float32) * 5.0,
            "loc": np.zeros((2, 3, 4), dtype=np.float32),
            "scale": np.ones((2, 3, 4), dtype=np.float32) * 2.0,
        }
    )

    # Input 7: Negative loc offset
    list_of_inputs.append(
        {
            "x": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "df": np.array([4.0, 4.0, 4.0], dtype=np.float64),
            "loc": np.array([-2.0, -1.0, -0.5], dtype=np.float64),
            "scale": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        }
    )

    # Input 8: Non-integer degrees of freedom
    list_of_inputs.append(
        {
            "x": np.array([0.5, 1.2, 3.7], dtype=np.float32),
            "df": np.array([0.5, 1.5, 2.7], dtype=np.float32),
            "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        }
    )

    # Input 9: Large degrees of freedom and large scale
    list_of_inputs.append(
        {
            "x": np.array([50.0, 100.0, 150.0], dtype=np.float64),
            "df": np.array([50.0, 100.0, 120.0], dtype=np.float64),
            "loc": np.array([10.0, 20.0, 30.0], dtype=np.float64),
            "scale": np.array([5.0, 10.0, 2.0], dtype=np.float64),
        }
    )

    # Input 10: Mixed dimensional broadcasting (3D x with 1D params)
    list_of_inputs.append(
        {
            "x": np.arange(1, 9, dtype=np.float32).reshape((2, 2, 2)),
            "df": np.array([2.0, 4.0], dtype=np.float32),
            "loc": np.array([0.0], dtype=np.float32),
            "scale": np.array([1.0], dtype=np.float32),
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.chi2.logpdf"] = chi2_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.chi2.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.chi2.logpdf'.")


check_valid('jax.scipy.stats.chi2.logpdf', generated_inputs['jax.scipy.stats.chi2.logpdf'], lib="jax", suffix=0)
