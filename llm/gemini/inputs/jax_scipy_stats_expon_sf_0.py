
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expon_sf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays, float64
    x = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float64)
    loc = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    scale = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: Values where x < loc (testing boundary condition)
    x = np.array([-1.0, 0.0, 0.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: 2D arrays of same shape
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    scale = np.array([[2.0, 1.0], [0.5, 3.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting - 2D x, 1D loc, 0D scale
    x = np.ones((3, 4), dtype=np.float32) * 2.0
    loc = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: Integer types
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    loc = np.array([0, 1, 1, 2], dtype=np.int32)
    scale = np.array([1, 2, 1, 3], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: 3D arrays
    x = np.random.uniform(2.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    loc = np.random.uniform(0.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    scale = np.random.uniform(0.5, 2.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Multidimensional broadcasting
    x = np.zeros((2, 1, 4), dtype=np.float64) + 3.0
    loc = np.zeros((1, 3, 1), dtype=np.float64) + 1.0
    scale = np.ones((2, 3, 4), dtype=np.float64) * 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Large values of x resulting in survival probabilities near 0
    x = np.array([10.0, 50.0, 100.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: 4D tensors
    x = np.full((1, 2, 2, 3), 4.0, dtype=np.float32)
    loc = np.full((1, 2, 2, 3), 1.0, dtype=np.float32)
    scale = np.full((1, 2, 2, 3), 2.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.expon.sf"] = expon_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.sf'.")


check_valid('jax.scipy.stats.expon.sf', generated_inputs['jax.scipy.stats.expon.sf'], lib="jax", suffix=0)
