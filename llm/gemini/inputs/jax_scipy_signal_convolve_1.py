
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_convolve_inputs():
    list_of_inputs = []

    # Signature A: 1D, shape (5,) and (3,), mode='full', method='auto', precision='default'
    # 1. Standard positive floats
    in1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    in2 = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. Negative floats
    in1 = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
    in2 = np.array([-1.0, -0.5, -0.2], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. Mixed signs
    in1 = np.array([-2.0, 1.5, -0.5, 3.0, -1.0], dtype=np.float32)
    in2 = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. Zero arrays
    in1 = np.zeros((5,), dtype=np.float32)
    in2 = np.zeros((3,), dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. One arrays
    in1 = np.ones((5,), dtype=np.float32)
    in2 = np.ones((3,), dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "full",
        "method": "auto",
        "precision": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Signature B: 2D, shape (3,3) and (2,2), mode='same', method='direct', precision='high'
    # 6. Random positive floats
    in1 = np.abs(np.random.randn(3, 3)).astype(np.float32)
    in2 = np.abs(np.random.randn(2, 2)).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": "high"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. Random negative floats
    in1 = -np.abs(np.random.randn(3, 3)).astype(np.float32)
    in2 = -np.abs(np.random.randn(2, 2)).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": "high"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. Mixed random floats
    in1 = np.random.randn(3, 3).astype(np.float32)
    in2 = np.random.randn(2, 2).astype(np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": "high"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. Integer-like floats
    in1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    in2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": "high"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. All ones (2D)
    in1 = np.ones((3, 3), dtype=np.float32)
    in2 = np.ones((2, 2), dtype=np.float32)
    input_dict = {
        "in1": in1,
        "in2": in2,
        "mode": "same",
        "method": "direct",
        "precision": "high"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.signal.convolve_1"] = generate_convolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.signal.convolve_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.signal.convolve_1'.")


check_valid('jax.scipy.signal.convolve', generated_inputs['jax.scipy.signal.convolve_1'], lib="jax", suffix=1)
