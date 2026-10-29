
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class Float(float):

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __bool__(self):
        return False

    def __hash__(self):
        return 0


def tan_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    accuracy = Float(0.0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 2D float32 array
    x = np.array([[-1.0, -0.25], [0.25, 1.0]], dtype=np.float32)
    accuracy = Float(0.0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 1D float64 array
    x = np.linspace(-1.2, 1.2, 10, dtype=np.float64)
    accuracy = Float(0.0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 2D float64 array
    x = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=np.float64)
    accuracy = Float(0.0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 3D float32 array
    x = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    accuracy = Float(0.0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 0D scalar float32 array
    x = np.array(0.42, dtype=np.float32)
    accuracy = Float(0.0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 4D float32 array
    x = np.random.uniform(-0.8, 0.8, size=(2, 2, 2, 2)).astype(np.float32)
    accuracy = Float(0.0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 1D complex64 array
    x = np.array([0.2 + 0.3j, -0.5 - 0.2j, 0.0 + 0.0j], dtype=np.complex64)
    accuracy = Float(0.0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 2D complex128 array
    x = np.array(
        [[0.1 + 0.5j, -0.2 + 0.1j], [0.3 - 0.4j, -0.5 - 0.5j]],
        dtype=np.complex128,
    )
    accuracy = Float(0.0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 1D float16 array
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float16)
    accuracy = Float(0.0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.tan_2"] = tan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tan_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tan_2'.")


check_valid('jax.lax.tan', generated_inputs['jax.lax.tan_2'], lib="jax", suffix=2)
