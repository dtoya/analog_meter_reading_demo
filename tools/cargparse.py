from argparse import ArgumentParser
import json
import os
from re import I

class ArgumentParser(ArgumentParser):
    def __init__(self, **kwargs):
        super(ArgumentParser, self).__init__(**kwargs)

    def parse_args(self, **kwargs):
        args_def = super(ArgumentParser, self).parse_args([])
        args = super(ArgumentParser, self).parse_args(**kwargs)
        if os.path.exists(args.config_file):
            config_file = open(args.config_file, 'r')
            config = json.load(config_file)
            for arg in config:
                if args.__dict__[arg] == args_def.__dict__[arg]:
                    args.__dict__[arg] = config[arg]   
        return args

    def dump(self, args, fp, **kwargs):    
        json.dump(args.__dict__, fp, **kwargs)


