import argparse
from datetime import datetime
from config import method_config, data_config


def _to_string(args, parser):
    ret = "main.py"
    for action in parser._get_positional_actions():
        ret += " " + str(getattr(args, action.dest))
    for action in parser._get_optional_actions():
        if action.default != "==SUPPRESS==":
            ret += " " + action.option_strings[0] + " " + str(getattr(args, action.dest))
    return ret


parser = argparse.ArgumentParser(description='python3 main.py "model" "dataset" "method" ')

# positional arguments
parser.add_argument("model", type=str, choices=['convnet','resnet18', 'resnet32', 'wide_resnet'],
                    help='specifies model')
parser.add_argument("dataset", type=str, choices=list(data_config.keys()),
                    help='specifies dataset')
parser.add_argument("method", type=str,
                    choices=list(method_config.keys()),
                    help='specifies training method')

# optional arguments
parser.add_argument("--desc", type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                    help='training description [default: cur_time]')
args = parser.parse_args()
command = _to_string(args, parser)
