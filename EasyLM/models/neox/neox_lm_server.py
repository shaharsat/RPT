
from EasyLM.models.neox.neox_serve import load_model
import mlxu
from EasyLM.serving import LMServer
import absl
absl.flags.DEFINE_multi_string(
'config_override', None, 'Newline separated list of model parameter overrides.')

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',
    dtype='bf16',
    input_length=1024,
    lowcoder_batch_size=8,
    seq_length=2048,
    top_k=50,
    top_p=1.0,
    do_sample=True,
    num_beams=1,
    add_bos_token=True,
    model_config_path='',
    load_checkpoint='',
    num_neighbors=0,
    keep_fields="",
    split_by_newline=False,
    port=35009,
    pre_compile="",
    lm_server_batch_size=1,
    return_empty=False,
)

import tempfile
import json
def main(argv):
    # print(FLAGS.append_flags_into_file("/tmp/flags.txt"))
    # FLAGS.serialize()
    
    flags = {k: {v.name: v.value for v in vs} for k, vs in FLAGS.flags_by_module_dict().items()}
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        with open(tfile.name, "w") as f:
            json.dump(flags, f)
        print(f"saved flags to {tfile.name}")
    
    
    _loglikelihood, _loglikelihood_rolling, _generate, _greedy_until, _encode, _old_greedy_until = load_model(FLAGS)
    class ModelServer(LMServer):

        @staticmethod
        def loglikelihood(prefix_text, text):
            return _loglikelihood(prefix_text, text)
            
        @staticmethod
        def loglikelihood_rolling(text):
            return _loglikelihood_rolling(text)
        
        @staticmethod
        def generate(text, temperature):
            return _generate(text, temperature)
        
        
        @staticmethod
        def old_greedy_until(prefix_text, until, max_length, pre_compile=False):
            return _old_greedy_until(prefix_text, until, max_length, pre_compile)
            
        @staticmethod
        def greedy_until(prefix_text, until, max_length, pre_compile=False):
            return _greedy_until(prefix_text, until, max_length, pre_compile)
        
        @staticmethod
        def encode(text):
            return _encode(text)
    
    server = ModelServer(LMServer.get_default_config(updates=dict(port=FLAGS.port,
                                                                  pre_compile=FLAGS.pre_compile,
                                                                  batch_size=FLAGS.lm_server_batch_size)))
    server.run()


if __name__ == "__main__":
    mlxu.run(main)
