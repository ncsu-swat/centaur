
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_real_inputs():
    list_of_inputs = []

    input_1 = tf.constant([1.0 + 2.0j, 3.0 - 4.0j], dtype=tf.complex64)
    input_dict_1 = {'input': input_1, 'Tout': tf.float32, 'name': 'real_op_1'}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = tf.constant([5.0 + 6.0j, 7.0 - 8.0j], dtype=tf.complex128)
    input_dict_2 = {'input': input_2, 'Tout': tf.float64, 'name': 'real_op_2'}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = tf.constant([[-1.0 + 2.0j, 3.0 - 4.0j], [5.0 + 6.0j, -7.0 - 8.0j]], dtype=tf.complex64)
    input_dict_3 = {'input': input_3, 'Tout': tf.float32, 'name': 'real_op_3'}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = tf.constant([1.0 + 2.0j, 3.0 - 4.0j], dtype=tf.complex128)
    input_dict_4 = {'input': input_4, 'Tout': tf.float64, 'name': 'real_op_4'}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = tf.constant([1.0 + 0.0j, 3.0 - 0.0j], dtype=tf.complex64)
    input_dict_5 = {'input': input_5, 'Tout': tf.float32, 'name': 'real_op_5'}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["tf.raw_ops.Real"] = tf_real_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Real'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Real', generated_inputs['tf.raw_ops.Real'], lib="tf", suffix=0)
