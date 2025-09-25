
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf

def tf_compute_accidental_hits_inputs():
    true_candidates = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.int64)
    sampled_candidates = tf.constant([[7, 8], [9, 10], [11, 12]], dtype=tf.int64)
    input_dict = {"args": [true_candidates, sampled_candidates]}
    return input_dict

generated_inputs["tf.raw_ops.ComputeAccidentalHits"] = tf_compute_accidental_hits_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ComputeAccidentalHits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ComputeAccidentalHits'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ComputeAccidentalHits', generated_inputs['tf.raw_ops.ComputeAccidentalHits'], lib="tf", suffix=0)
