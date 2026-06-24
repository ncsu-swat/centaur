
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def xlog1py_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar-like tensors, float32
    x = np.array(2.5, dtype=np.float32)
    y = np.array(4.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 1D arrays, float32, standard range
    x = np.array([0.0, 1.5, 3.0], dtype=np.float32)
    y = np.array([0.5, 2.0, 4.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: Special case where x = 0 and y = -1 (defined to return 0)
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 2D arrays, float64, positive values
    x = np.random.uniform(0.1, 10.0, (3, 4)).astype(np.float64)
    y = np.random.uniform(0.1, 10.0, (3, 4)).astype(np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: Broadcastable shapes (3, 1) and (1, 4)
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 3D arrays with negative x and positive y
    x = np.random.uniform(-5.0, 0.0, (2, 2, 3)).astype(np.float32)
    y = np.random.uniform(0.0, 5.0, (2, 2, 3)).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Small y values (tests precision log(1+y) ~ y)
    x = np.array([10.0, 100.0, 1000.0], dtype=np.float64)
    y = np.array([1e-8, 1e-7, 1e-6], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: All zeros for x, arbitrary negative values for y (should all return 0)
    x = np.zeros((4, 2), dtype=np.float32)
    y = np.random.uniform(-0.99, -0.1, (4, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Large multi-dimensional array with float64
    x = np.random.uniform(1.0, 100.0, (2, 3, 4, 2)).astype(np.float64)
    y = np.random.uniform(-0.5, 50.0, (2, 3, 4, 2)).astype(np.float64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: Complex broadcasting with higher dimensions
    x = np.random.uniform(0.1, 5.0, (1, 4, 1, 3)).astype(np.float32)
    y = np.random.uniform(0.1, 5.0, (2, 1, 3, 1)).astype(np.float32)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.scipy.special.xlog1py"] = xlog1py_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.xlog1py' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.xlog1py'.")


check_valid('jax.scipy.special.xlog1py', generated_inputs['jax.scipy.special.xlog1py'], lib="jax", suffix=0)
