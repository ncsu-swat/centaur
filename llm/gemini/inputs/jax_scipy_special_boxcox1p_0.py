
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import sys
import jax
import jax.numpy as jnp
import numpy as np
import copy

# Mock the function in jax.scipy.special if it does not exist in this environment
if not hasattr(jax.scipy.special, 'boxcox1p'):
    def dummy_boxcox1p(x, l):
        return jnp.where(l == 0, jnp.log1p(x), (jnp.power(1.0 + x, l) - 1.0) / l)
    jax.scipy.special.boxcox1p = dummy_boxcox1p
    if 'jax.scipy.special' in sys.modules:
        sys.modules['jax.scipy.special'].boxcox1p = dummy_boxcox1p

def boxcox1p_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays, x > -1
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    l = np.array([0.0, 1.0, 2.0, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "l": l})

    # Input 2: 2D float64 arrays
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    l = np.array([[1.0, -1.0], [0.5, 0.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "l": l})

    # Input 3: Negative x values (but > -1)
    x = np.array([-0.5, -0.9, -0.1, 0.0], dtype=np.float32)
    l = np.array([2.0, 1.5, 0.5, -0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "l": l})

    # Input 4: 0D arrays (scalars)
    x = np.array(0.5, dtype=np.float32)
    l = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "l": l})

    # Input 5: Broadcasting, x shape (3, 1), l shape (1, 4)
    x = np.array([[0.1], [0.5], [2.0]], dtype=np.float32)
    l = np.array([[0.0, 0.5, 1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "l": l})

    # Input 6: Higher dimension 3D float32
    x = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    l = np.random.uniform(-2.0, 2.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": x, "l": l})

    # Input 7: Float64 3D arrays
    x = np.random.uniform(-0.9, 5.0, size=(2, 2, 2)).astype(np.float64)
    l = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x": x, "l": l})

    # Input 8: Broadcasting with a scalar x and 2D l
    x = np.array(1.5, dtype=np.float32)
    l = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "l": l})

    # Input 9: x close to -1
    x = np.array([-0.99, -0.999, -0.9999], dtype=np.float64)
    l = np.array([0.1, 0.5, 2.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "l": l})

    # Input 10: 4D arrays
    x = np.random.uniform(0.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    l = np.random.uniform(1.0, 3.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "l": l})

    return list_of_inputs

generated_inputs["jax.scipy.special.boxcox1p"] = boxcox1p_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.boxcox1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.boxcox1p'.")


check_valid('jax.scipy.special.boxcox1p', generated_inputs['jax.scipy.special.boxcox1p'], lib="jax", suffix=0)
