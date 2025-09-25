
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_compat_path_to_str_inputs():
    list_of_inputs = []
    path1 = tf.constant("C:\\XYZ\\tensorflow\\./.././tensorflow")
    path2 = tf.constant("C:\XYZ\tensorflow\./.././tensorflow")
    path3 = tf.constant("./corpus")
    path4 = tf.constant("./.././Corpus")
    path5 = tf.constant("./.././Corpus")
    path6 = tf.constant("./..////../")
    path7 = tf.constant("path/to/file")
    path8 = tf.constant("another_path")
    path9 = tf.constant("path/with/../relative/path")
    path10 = tf.constant("C:/XYZ/tensorflow/./.././tensorflow")
    path11 = tf.constant("C:\\XYZ\\tensorflow\\./.././tensorflow")
    path12 = tf.constant("Relative/Path/Test")

    input_dict1 = {"path": path1}
    input_dict2 = {"path": path2}
    input_dict3 = {"path": path3}
    input_dict4 = {"path": path4}
    input_dict5 = {"path": path5}
    input_dict6 = {"path": path6}
    input_dict7 = {"path": path7}
    input_dict8 = {"path": path8}
    input_dict9 = {"path": path9}
    input_dict10 = {"path": path10}
    input_dict11 = {"path": path11}
    input_dict12 = {"path": path12}

    list_of_inputs.append(copy.deepcopy(input_dict1))
    list_of_inputs.append(copy.deepcopy(input_dict2))
    list_of_inputs.append(copy.deepcopy(input_dict3))
    list_of_inputs.append(copy.deepcopy(input_dict4))
    list_of_inputs.append(copy.deepcopy(input_dict5))
    list_of_inputs.append(copy.deepcopy(input_dict6))
    list_of_inputs.append(copy.deepcopy(input_dict7))
    list_of_inputs.append(copy.deepcopy(input_dict8))
    list_of_inputs.append(copy.deepcopy(input_dict9))
    list_of_inputs.append(copy.deepcopy(input_dict10))
    list_of_inputs.append(copy.deepcopy(input_dict11))
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs["tf.compat.path_to_str"] = tf_compat_path_to_str_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.compat.path_to_str' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.compat.path_to_str'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.compat.path_to_str', generated_inputs['tf.compat.path_to_str'], lib="tf", suffix=0)
