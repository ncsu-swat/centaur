
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tile_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, reps = 2
    A = np.array([1, 2, 3], dtype=np.int32)
    reps = 2
    list_of_inputs.append({"A": A, "reps": reps})

    # Input 2: 2D float32 array, reps = 3
    A = np.random.randn(2, 3).astype(np.float32)
    reps = 3
    list_of_inputs.append({"A": A, "reps": reps})

    # Input 3: 3D float64 array with negative values, reps = 1
    A = np.random.uniform(-10.0, 10.0, size=(2, 2, 2)).astype(np.float64)
    reps = 1
    list_of_inputs.append({"A": A, "reps": reps})

    # Input 4: 1D float32 array, reps = 0
    A = np.array([0.5, -0.5, 1.5], dtype=np.float32)
    reps = 0
    list_of_inputs.append({"A": A, "reps": reps})

    # Input 5: 4D int8 array, reps = 2
    A = np.random.randint(-5, 5, size=(1, 2, 1, 3)).astype(np.int8)
    reps = 2
    list_of_inputs.append({"A": A, "reps": reps})

    # Input 6: 2D boolean array, reps = 4
    A = np.array([[True, False], [False, True]], dtype=bool)
    reps = 4
    list_of_inputs.append({"A": A, "reps": reps})

    # Input 7: 1D complex array, reps = 2
    A = np.array([1 + 2j, 3 - 4j], dtype=np.complex64)
    reps = 2
    list_of_inputs.append({"A": A, "reps": reps})

    # Input 8: 0D array (scalar), reps = 3
    A = np.array(42)
    reps = 3
    list_of_inputs.append({"A": A, "reps": reps})

    # Input 9: Large 2D float32 array, reps = 5
    A = np.random.randn(10, 10).astype(np.float32)
    reps = 5
    list_of_inputs.append({"A": A, "reps": reps})

    # Input 10: 3D int16 array, reps = 10
    A = np.random.randint(0, 100, size=(2, 1, 3)).astype(np.int16)
    reps = 10
    list_of_inputs.append({"A": A, "reps": reps})

    # Input 11: 1D uint8 array, reps = 1
    A = np.array([255, 128, 0], dtype=np.uint8)
    reps = 1
    list_of_inputs.append({"A": A, "reps": reps})

    return [copy.deepcopy(x) for x in list_of_inputs]

generated_inputs["jax.numpy.tile_1"] = tile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tile_1'.")


check_valid('jax.numpy.tile', generated_inputs['jax.numpy.tile_1'], lib="jax", suffix=1)
