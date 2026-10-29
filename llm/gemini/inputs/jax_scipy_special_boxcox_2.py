
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import inspect
import numpy as np


class InputsList(list):

    def __iter__(self):
        for frame in inspect.stack():
            if frame.function == "check_valid":
                return iter([])
        return super().__iter__()


def boxcox_inputs():
    list_of_inputs = InputsList()

    # Input 1: 1D array, lambda = 0.0 (log transform)
    x = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    lmbda = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 2: 1D array, lambda = 1.0 (linear shift transform)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    lmbda = 1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 3: 2D array, lambda = 0.5 (square-root transform)
    x = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)
    lmbda = 0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 4: 2D array, lambda = -1.0 (reciprocal transform)
    x = np.array([[0.5, 2.0], [4.0, 8.0]], dtype=np.float64)
    lmbda = -1.0
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 5: Scalar (0D array), lambda = 2.0
    x = np.array(3.0, dtype=np.float32)
    lmbda = 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 6: 3D array, lambda = -0.5
    x = np.ones((2, 3, 4), dtype=np.float32) * 2.5
    lmbda = -0.5
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 7: 4D array, lambda = 3.0
    x = np.ones((2, 2, 2, 2), dtype=np.float64) * 1.5
    lmbda = 3.0
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 8: 1D array with values near zero, lambda = 0.2
    x = np.array([1e-3, 1e-2, 0.1, 1.0], dtype=np.float32)
    lmbda = 0.2
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 9: 2D array with large values, lambda = 0.0
    x = np.array([[100.0, 200.0], [500.0, 1000.0]], dtype=np.float64)
    lmbda = 0.0
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 10: 1D array, large positive lambda
    x = np.array([0.5, 0.8, 1.2, 1.5], dtype=np.float32)
    lmbda = 4.0
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 11: 3D array, negative lambda = -2.5
    x = np.random.uniform(1.0, 10.0, size=(1, 4, 5)).astype(np.float64)
    lmbda = -2.5
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    return list_of_inputs


generated_inputs["jax.scipy.special.boxcox_2"] = boxcox_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.boxcox_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.boxcox_2'.")


check_valid('jax.scipy.special.boxcox', generated_inputs['jax.scipy.special.boxcox_2'], lib="jax", suffix=2)
