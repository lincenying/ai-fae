import os
import sys
import mindspore as ms

def prepend_git_root_dir_to_python_path():
    """Prepend root folder of current git repo to Python's sys.path.

    This would force Python use package 'mindformers' in this project
    rather than the one installed in folder site-packages."""
    
    import subprocess
    try:
        ret = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE)
        git_dir = ret.stdout.decode('utf-8').strip()
    except:
        git_dir = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
    print("Prepending '%s' to PYTHONPATH" % git_dir)
    sys.path.insert(0, git_dir)

    qwen_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
    print("Prepending '%s' to PYTHONPATH" % qwen_dir)
    sys.path.insert(0, qwen_dir)

    
def load_model_and_tokenizer(args, use_past=False):
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=args.device_id)
    # ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    
    from mindformers.models.llama.llama_config import LlamaConfig
    from mindformers.tools.register.config import MindFormerConfig
    from qwen_config import QwenConfig
    from qwen_model import QwenForCausalLM
    from qwen_tokenizer import QwenTokenizer, ENDOFTEXT

    print("Initializing with configrartioin file %s" % args.config)
    config = MindFormerConfig(args.config)
    
    tokenizer_config = config.processor.tokenizer
    if args.vocab_file:
        tokenizer_config.vocab_file = args.vocab_file
    
    tokenizer_config.model_max_length = args.seq_length
    
    tokenizer = QwenTokenizer(**tokenizer_config)

    model_config = QwenConfig.from_pretrained(args.config)
    if args.checkpoint_path:
        model_config.checkpoint_name_or_path = args.checkpoint_path


    model_config.seq_length = args.seq_length
    model_config.max_position_embedding = args.seq_length
    model_config.max_length = args.seq_length

    model_config.use_past = use_past

    model = QwenForCausalLM(model_config)

    return model, tokenizer


def add_argparse_common_args(parser):
    from distutils.util import strtobool
    
    group = parser.add_argument_group(title="Common options")

    group.add_argument('--config', default='run_qwen_7b.yaml',
                       type=str, help='Config file path. (default: ./run_qwen_7b.yaml)')

    group.add_argument("--checkpoint_path", default="",
                       type=str, help="Checkpoint path.")
    group.add_argument('--vocab_file', default="",
                       type=str, help='Tokenizer model.')

    parser.add_argument('--device_id', default=0, type=int,
                        help='ID of the target device, the value must be in [0, device_num_per_host-1]')
    
    group.add_argument("--seq_length", default=2048,
                       type=int, help="model.seq_length (default: 2048)")
    
    group.add_argument("-s", "--seed", default=1234,
                       type=int, help="Random seed. (default: 1234)")    
    group.add_argument("--debug", action="store_true", default=False,
                       help="Print infos.")
    
