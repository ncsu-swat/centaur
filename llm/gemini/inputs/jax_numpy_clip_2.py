
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def clip_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with symmetric bounds
    arr = np.array([-3.0, -1.0, 0.0, 1.5, 3.0], dtype=np.float32)
    input_dict = {"arr": arr, "min": -1.0, "max": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array with positive bounds
    arr = np.array([[0.5, 2.3, 7.1], [4.4, -1.2, 3.0]], dtype=np.float64)
    input_dict = {"arr": arr, "min": 0.0, "max": 5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array
    arr = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    input_dict = {"arr": arr, "min": 5.0, "max": 15.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int64 array with negative bounds
    arr = np.array([-50, -20, -5, 0, 10, 30], dtype=np.int64)
    input_dict = {"arr": arr, "min": -30.0, "max": -10.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array
    arr = np.random.randn(2, 2, 3, 4).astype(np.float32)
    input_dict = {"arr": arr, "min": -0.5, "max": 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D scalar float32 array
    arr = np.array(4.5, dtype=np.float32)
    input_dict = {"arr": arr, "min": 1.0, "max": 3.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 array with identical min and max
    arr = np.random.uniform(-10.0, 10.0, size=(4, 4)).astype(np.float32)
    input_dict = {"arr": arr, "min": 0.0, "max": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float16 array
    arr = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float16)
    input_dict = {"arr": arr, "min": -0.25, "max": 0.75}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D int16 array
    arr = np.random.randint(-100, 100, size=(2, 1, 2, 2, 3), dtype=np.int16)
    input_dict = {"arr": arr, "min": -25.0, "max": 25.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float64 array with large range bounds
    arr = np.linspace(-1000.0, 1000.0, 27, dtype=np.float64).reshape(3, 3, 3)
    input_dict = {"arr": arr, "min": -100.0, "max": 500.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D uint8 array
    arr = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.uint8)
    input_dict = {"arr": arr, "min": 30.0, "max": 180.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.clip_2"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.clip_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.clip_2'.")


check_valid('jax.numpy.clip', generated_inputs['jax.numpy.clip_2'], lib="jax", suffix=2)
