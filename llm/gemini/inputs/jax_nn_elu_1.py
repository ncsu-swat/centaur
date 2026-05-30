
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_nn_elu_inputs():
    list_of_inputs = []

    # Input 1: 1D array, standard float32, alpha = 1.0
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    alpha = 1.0
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 2: 2D array, float32, alpha = 0.5
    x = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    alpha = 0.5
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 3: 3D array, float64, alpha = 2.0
    x = np.random.randn(2, 3, 4).astype(np.float64)
    alpha = 2.0
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 4: 1D array, all negative, float64, alpha = 1.5
    x = np.array([-10.0, -5.0, -1.0, -0.1], dtype=np.float64)
    alpha = 1.5
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 5: 4D array, float32, alpha = 0.1
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    alpha = 0.1
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 6: 0D array (scalar tensor), float32, alpha = 1.0
    x = np.array(-1.5, dtype=np.float32)
    alpha = 1.0
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 7: 2D array, large positive and negative values, float32, alpha = 5.0
    x = np.array([[-100.0, 100.0], [-50.0, 50.0]], dtype=np.float32)
    alpha = 5.0
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 8: 1D array, all zeros, float32, alpha = 0.0
    x = np.zeros((5,), dtype=np.float32)
    alpha = 0.0
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 9: 3D array, positive values only, float32, alpha = 0.8
    x = np.abs(np.random.randn(2, 2, 2).astype(np.float32)) + 0.1
    alpha = 0.8
    list_of_inputs.append({"x": x, "alpha": alpha})

    # Input 10: 2D array, very small values close to zero, float64, alpha = 1.2
    x = np.array([[-1e-5, 1e-5], [-1e-6, 1e-6]], dtype=np.float64)
    alpha = 1.2
    list_of_inputs.append({"x": x, "alpha": alpha})

    return list_of_inputs

generated_inputs["jax.nn.elu_1"] = jax_nn_elu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.elu_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.elu_1'.")


check_valid('jax.nn.elu', generated_inputs['jax.nn.elu_1'], lib="jax", suffix=1)
