
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_image_adjust_saturation_inputs():
    list_of_inputs = []
    image1 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=tf.float32)
    saturation_factor1 = 0.5
    input_dict1 = {"image": image1, "saturation_factor": saturation_factor1, "name": "adjust_sat1"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    image2 = tf.constant([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]], dtype=tf.float32)
    saturation_factor2 = 2.0
    input_dict2 = {"image": image2, "saturation_factor": saturation_factor2, "name": "adjust_sat2"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    image3 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=tf.float32)
    saturation_factor3 = -1.0
    input_dict3 = {"image": image3, "saturation_factor": saturation_factor3, "name": "adjust_sat3"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    image4 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=tf.float64)
    saturation_factor4 = 1.5
    input_dict4 = {"image": image4, "saturation_factor": saturation_factor4, "name": "adjust_sat4"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    image5 = tf.zeros((10, 10, 3), dtype=tf.float32)
    saturation_factor5 = 0.0
    input_dict5 = {"image": image5, "saturation_factor": saturation_factor5, "name": "adjust_sat5"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    image6 = tf.ones((5, 5, 3), dtype=tf.float32)
    saturation_factor6 = 1.0
    input_dict6 = {"image": image6, "saturation_factor": saturation_factor6, "name": "adjust_sat6"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    image7 = tf.random.uniform((2, 2, 3), minval=0.0, maxval=1.0, dtype=tf.float32)
    saturation_factor7 = 0.75
    input_dict7 = {"image": image7, "saturation_factor": saturation_factor7, "name": "adjust_sat7"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    image8 = tf.constant([[[1.0, 2.0, 3.0]]], dtype=tf.float32)
    saturation_factor8 = 2.5
    input_dict8 = {"image": image8, "saturation_factor": saturation_factor8, "name": "adjust_sat8"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    image9 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=tf.float64)
    saturation_factor9 = -0.5
    input_dict9 = {"image": image9, "saturation_factor": saturation_factor9, "name": "adjust_sat9"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    image10 = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=tf.float32)
    saturation_factor10 = 3.0
    input_dict10 = {"image": image10, "saturation_factor": saturation_factor10, "name": "adjust_sat10"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.image.adjust_saturation"] = tf_image_adjust_saturation_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_saturation' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_saturation'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_saturation', generated_inputs['tf.image.adjust_saturation'], lib="tf", suffix=0)
