
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nbinom_pmf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    list_of_inputs.append(
        {
            "k": np.array(5.0, dtype=np.float32),
            "n": np.array(10.0, dtype=np.float32),
            "p": np.array(0.5, dtype=np.float32),
            "loc": np.array(0.0, dtype=np.float32),
        }
    )

    # Input 2: 1D arrays, same shapes, float64
    list_of_inputs.append(
        {
            "k": np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            "n": np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64),
            "p": np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float64),
            "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        }
    )

    # Input 3: Non-zero loc
    list_of_inputs.append(
        {
            "k": np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
            "n": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
            "p": np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float32),
            "loc": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        }
    )

    # Input 4: 2D arrays, broadcastable shapes
    list_of_inputs.append(
        {
            "k": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            "n": np.array([[4.0, 5.0]], dtype=np.float32),
            "p": np.array([[0.2, 0.8]], dtype=np.float32),
            "loc": np.array([[0.0]], dtype=np.float32),
        }
    )

    # Input 5: 2D arrays of matching shape (3, 3)
    list_of_inputs.append(
        {
            "k": np.arange(9, dtype=np.float32).reshape(3, 3),
            "n": np.full((3, 3), 10.0, dtype=np.float32),
            "p": np.full((3, 3), 0.6, dtype=np.float32),
            "loc": np.zeros((3, 3), dtype=np.float32),
        }
    )

    # Input 6: 3D arrays
    list_of_inputs.append(
        {
            "k": np.ones((2, 3, 4), dtype=np.float32) * 5,
            "n": np.ones((2, 3, 4), dtype=np.float32) * 3,
            "p": np.ones((2, 3, 4), dtype=np.float32) * 0.7,
            "loc": np.zeros((2, 3, 4), dtype=np.float32),
        }
    )

    # Input 7: Broadcasting scalar loc to 1D inputs
    list_of_inputs.append(
        {
            "k": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            "n": np.array([15.0, 15.0, 15.0], dtype=np.float32),
            "p": np.array([0.5, 0.5, 0.5], dtype=np.float32),
            "loc": np.array(5.0, dtype=np.float32),
        }
    )

    # Input 8: Varying parameters across batch
    list_of_inputs.append(
        {
            "k": np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32),
            "n": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "p": np.array([0.1, 0.25, 0.5, 0.75], dtype=np.float32),
            "loc": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        }
    )

    # Input 9: Large values for k and n
    list_of_inputs.append(
        {
            "k": np.array([100.0, 150.0], dtype=np.float64),
            "n": np.array([50.0, 75.0], dtype=np.float64),
            "p": np.array([0.35, 0.45], dtype=np.float64),
            "loc": np.array([0.0, 0.0], dtype=np.float64),
        }
    )

    # Input 10: 4D tensors
    list_of_inputs.append(
        {
            "k": np.ones((2, 2, 2, 2), dtype=np.float32) * 4,
            "n": np.ones((2, 2, 2, 2), dtype=np.float32) * 6,
            "p": np.ones((2, 2, 2, 2), dtype=np.float32) * 0.4,
            "loc": np.zeros((2, 2, 2, 2), dtype=np.float32),
        }
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.nbinom.pmf"] = nbinom_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.nbinom.pmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.nbinom.pmf'.")


check_valid('jax.scipy.stats.nbinom.pmf', generated_inputs['jax.scipy.stats.nbinom.pmf'], lib="jax", suffix=0)
