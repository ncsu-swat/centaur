
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def elu_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default-like alpha
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    alpha = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 2: 2D array, standard normal, float32
    x = np.random.randn(3, 4).astype(np.float32)
    alpha = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 3: 3D array, float64, alpha = 2.0
    x = np.random.randn(2, 3, 4).astype(np.float64)
    alpha = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 4: Scalar 0D array
    x = np.array(-0.75, dtype=np.float32)
    alpha = 1.67
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 5: All positive values
    x = np.linspace(0.1, 10.0, num=15, dtype=np.float32).reshape(3, 5)
    alpha = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 6: All negative values
    x = np.linspace(-10.0, -0.1, num=12, dtype=np.float32).reshape(2, 2, 3)
    alpha = 0.1
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 7: 4D tensor, float16
    x = np.random.uniform(-3.0, 3.0, size=(2, 2, 4, 4)).astype(np.float16)
    alpha = 1.5
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 8: Empty array (0 shape dimension)
    x = np.empty((0, 5), dtype=np.float32)
    alpha = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 9: Extreme negative values to test saturation
    x = np.array([-100.0, -50.0, -20.0, 0.0], dtype=np.float32)
    alpha = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 10: 5D array, float32, alpha = 0.0
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    alpha = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    # Input 11: Large alpha value
    x = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    alpha = 100.0
    list_of_inputs.append(copy.deepcopy({"x": x, "alpha": alpha}))

    return list_of_inputs


generated_inputs["jax.nn.elu_1"] = elu_inputs()

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
