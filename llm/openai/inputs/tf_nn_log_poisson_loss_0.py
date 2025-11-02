
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def tf_nn_log_poisson_loss_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    targets = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    log_input = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    compute_full_loss = False
    name = "test1"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 2, valid
    targets = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    log_input = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    compute_full_loss = True
    name = "test2"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 3, valid
    targets = np.array([1, 2, 3, 4], dtype=np.float32)
    log_input = np.array([1, 2, 3, 4], dtype=np.float32)
    compute_full_loss = False
    name = "test3"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 4, valid
    targets = np.array([-1, -2, -3], dtype=np.float32)
    log_input = np.array([-1, -2, -3], dtype=np.float32)
    compute_full_loss = False
    name = "test4"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 5, valid
    targets = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    log_input = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    compute_full_loss = True
    name = "test5"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 6, valid
    targets = np.array([[1, 2], [3, 4]], dtype=np.float32)
    log_input = np.array([[1, 2], [3, 4]], dtype=np.float32)
    compute_full_loss = True
    name = "test6"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 7, valid
    targets = np.array([1.5, 2.7, 3.1], dtype=np.float32)
    log_input = np.array([1.5, 2.7, 3.1], dtype=np.float32)
    compute_full_loss = False
    name = "test7"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 8, valid
    targets = np.array([0, 1, 2], dtype=np.float32)
    log_input = np.array([0, 1, 2], dtype=np.float32)
    compute_full_loss = True
    name = "test8"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 9, valid
    targets = np.array([2.3, 3.4, 4.5], dtype=np.float32)
    log_input = np.array([2.3, 3.4, 4.5], dtype=np.float32)
    compute_full_loss = True
    name = "test9"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    # Input 10, valid
    targets = np.array([5.6, 6.7, 7.8], dtype=np.float32)
    log_input = np.array([5.6, 6.7, 7.8], dtype=np.float32)
    compute_full_loss = False
    name = "test10"
    
    input_dict = {
        "targets": tf.constant(targets),
        "log_input": tf.constant(log_input),
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(input_dict.copy())
    
    return list_of_inputs

generated_inputs["tf.nn.log_poisson_loss"] = tf_nn_log_poisson_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.log_poisson_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.log_poisson_loss'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.log_poisson_loss', generated_inputs['tf.nn.log_poisson_loss'], lib="tf", suffix=0)
