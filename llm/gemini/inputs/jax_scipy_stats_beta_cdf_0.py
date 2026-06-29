
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def beta_cdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32, standard range [0, 1]
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    a = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {'x': x, 'a': a, 'b': b, 'loc': loc, 'scale': scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D scalar tensors (float64) with non-standard loc and scale
    x = np.array(1.5, dtype=np.float64)
    a = np.array(2.0, dtype=np.float64)
    b = np.array(2.0, dtype=np.float64)
    loc = np.array(1.0, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    input_dict = {'x': x, 'a': a, 'b': b, 'loc': loc, 'scale': scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float32, random valid values
    x = np.random.uniform(0.1, 0.9, size=(2, 3)).astype(np.float32)
    a = np.array([[1.5, 2.5, 3.5], [1.5, 2.5, 3.5]], dtype=np.float32)
    b = np.array([[3.5, 2.5, 1.5], [3.5, 2.5, 1.5]], dtype=np.float32)
    loc = np.zeros((2, 3), dtype=np.float32)
    scale = np.ones((2, 3), dtype=np.float32)
    input_dict = {'x': x, 'a': a, 'b': b, 'loc': loc, 'scale': scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D arrays, float32, checking scaled values
    x = np.random.uniform(1.0, 5.0, size=(2, 2, 2)).astype(np.float32)
    a = np.full((2, 2, 2), 2.0, dtype=np.float32)
    b = np.full((2, 2, 2), 5.0, dtype=np.float32)
    loc = np.full((2, 2, 2), 1.0, dtype=np.float32)
    scale = np.full((2, 2, 2), 4.0, dtype=np.float32)
    input_dict = {'x': x, 'a': a, 'b': b, 'loc': loc, 'scale': scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x values strictly below loc (should yield CDF = 0)
    x = np.array([-1.0, -0.5], dtype=np.float32)
    a = np.array([1.0, 1.0], dtype=np.float32)
    b = np.array([1.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {'x': x, 'a': a, 'b': b, 'loc': loc, 'scale': scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x values strictly above loc + scale (should yield CDF = 1)
    x = np.array([2.0, 3.0], dtype=np.float32)
    a = np.array([1.0, 1.0], dtype=np.float32)
    b = np.array([1.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {'x': x, 'a': a, 'b': b, 'loc': loc, 'scale': scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting behavior with varied shapes
    x = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)  # shape (2, 2)
    a = np.array([1.0, 2.0], dtype=np.float32)               # shape (2,)
    b = np.array([[3.0], [4.0]], dtype=np.float32)            # shape (2, 1)
    loc = np.array(0.0, dtype=np.float32)                     # shape ()
    scale = np.array([1.0], dtype=np.float32)                 # shape (1,)
    input_dict = {'x': x, 'a': a, 'b': b, 'loc': loc, 'scale': scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: U-shaped beta distribution shape parameters (a, b < 1)
    x = np.array([0.1, 0.9], dtype=np.float32)
    a = np.array([0.5, 0.5], dtype=np.float32)
    b = np.array([0.5, 0.5], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {'x': x, 'a': a, 'b': b, 'loc': loc, 'scale': scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large shape parameters (bell-shaped beta)
    x = np.array([5.0, 6.0], dtype=np.float32)
    a = np.array([10.0, 20.0], dtype=np.float32)
    b = np.array([10.0, 20.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([10.0, 10.0], dtype=np.float32)
    input_dict = {'x': x, 'a': a, 'b': b, 'loc': loc, 'scale': scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Float64 high-precision inputs
    x = np.array([0.123456789012345], dtype=np.float64)
    a = np.array([2.123456789012345], dtype=np.float64)
    b = np.array([3.123456789012345], dtype=np.float64)
    loc = np.array([0.0], dtype=np.float64)
    scale = np.array([1.0], dtype=np.float64)
    input_dict = {'x': x, 'a': a, 'b': b, 'loc': loc, 'scale': scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.beta.cdf"] = beta_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.beta.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.beta.cdf'.")


check_valid('jax.scipy.stats.beta.cdf', generated_inputs['jax.scipy.stats.beta.cdf'], lib="jax", suffix=0)
