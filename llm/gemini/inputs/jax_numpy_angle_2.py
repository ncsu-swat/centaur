
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def angle_inputs():
    list_of_inputs = []

    # Input 1: Float32 scalar, deg=False
    z = np.array(1.5, dtype=np.float32)
    deg = False
    list_of_inputs.append({"z": z, "deg": deg})

    # Input 2: Float32 scalar negative, deg=True
    z = np.array(-2.5, dtype=np.float32)
    deg = True
    list_of_inputs.append({"z": z, "deg": deg})

    # Input 3: 1D Float32 array, deg=False
    z = np.array([1.0, -2.0, 0.0], dtype=np.float32)
    deg = False
    list_of_inputs.append({"z": z, "deg": deg})

    # Input 4: 2D Float64 array, deg=True
    z = np.array([[0.5, -1.5], [2.5, -3.5]], dtype=np.float64)
    deg = True
    list_of_inputs.append({"z": z, "deg": deg})

    # Input 5: Float32 scalar zero, deg=False
    z = np.array(0.0, dtype=np.float32)
    deg = False
    list_of_inputs.append({"z": z, "deg": deg})

    # Input 6: 1D Float64 array, deg=True
    z = np.array([-0.001, 0.002, -3.14], dtype=np.float64)
    deg = True
    list_of_inputs.append({"z": z, "deg": deg})

    # Input 7: 3D Float32 array, deg=False
    z = np.array([[[1.1, -2.2]], [[3.3, -4.4]]], dtype=np.float32)
    deg = False
    list_of_inputs.append({"z": z, "deg": deg})

    # Input 8: numpy float32 scalar, deg=True
    z = np.float32(10.0)
    deg = True
    list_of_inputs.append({"z": z, "deg": deg})

    # Input 9: numpy float64 scalar, deg=False
    z = np.float64(-100.0)
    deg = False
    list_of_inputs.append({"z": z, "deg": deg})

    # Input 10: 1D Float32 array with negative/positive zeros, deg=True
    z = np.array([-0.0, 0.0], dtype=np.float32)
    deg = True
    list_of_inputs.append({"z": z, "deg": deg})

    return list_of_inputs

generated_inputs["jax.numpy.angle_2"] = angle_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.angle_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.angle_2'.")


check_valid('jax.numpy.angle', generated_inputs['jax.numpy.angle_2'], lib="jax", suffix=2)
