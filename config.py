import yaml
import json

class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=obj)

def parse(fname):
    global args, states
    defaults = {'T': 6,
                'tau': 2.0,
                'b': 128, # batch size
                'epochs': 300,
                'j': 8, # number of data loading workers (default: 8)
                'resume': None, # resume from the checkpoint path
                'amp': False,
                'opt': 'SGD', # optimizer, SGD or Adam
                'lr': 0.1,
                'momentum': 0.9, # momentum for SGD
                'lr_scheduler': 'CosALR', # StepLR or CosALR
                'step_size': 100, # step_size for StepLR
                'gamma': 0.1, # gamma for StepLR
                'T_max': 300, # T_max for CosineAnnealingLR
                'model': 'online_spiking_vgg11_ws',
                'drop_rate': 0.0,
                'stochdepth_rate': 0.0,
                'weight_decay': 0.0,
                'cnf': '',
                'T_train': None,
                'loss_lambda': 0.05,
                'online_update': False,
                'BN': False,
                'WS': True,
                'BPTT': False,
                'tau_online_level': 1, # online level of tau, 1 for baseline, 5 for max online level
                'weight_online_level': 1, # online level of weight, 1 for baseline, 4 for max online level
                }
    require_args = {'data_dir',
                    'dataset',
                    # 'out_dir',
                    }
    
    with open(fname, 'r') as f:
        file = f.read()
    args = yaml.safe_load(file)
    args = dict2obj(args)
    for key in defaults:
        if not hasattr(args, key):
            setattr(args, key, defaults[key])
    for key in require_args:
        assert(hasattr(args, key))
    args.dataset = args.dataset.lower()
    args.tau = float(args.tau)
    if args.T_train is None:
        args.T_train = args.T
    
    states = dict2obj({'T': args.T})