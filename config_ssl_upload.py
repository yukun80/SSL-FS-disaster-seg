"""Sacred experiment configuration for the remote-sensing few-shot pipeline."""
import os
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('mySSL')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)


def _load_id_list(path: str | None) -> list[str] | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as handle:
        return [line.strip() for line in handle if line.strip()]

@ex.config
def cfg():
    """Default configurations"""
    seed = 1234
    gpu_id = 0
    mode = 'train' # for now only allows 'train' 
    do_validation=False
    num_workers = 0 # 0 for debugging. 

    dataset = 'POTSDAM_BIJIE'  # remote-sensing landslide dataset

    # Training data setup -------------------------------------------------
    n_steps = 100100
    batch_size = 1
    lr_milestones = [(ii + 1) * 1000 for ii in range(n_steps // 1000 - 1)]
    lr_step_gamma = 0.95
    ignore_label = 255
    print_interval = 100
    save_snapshot_every = 25000
    max_iters_per_load = 1000  # episode count per synthetic epoch
    epochs = 1
    scan_per_load = -1
    which_aug = 'satellite_aug'
    input_size = (512, 512)
    usealign = True
    use_wce = True
    grad_accumulation_steps = 1

    # Validation / evaluation -------------------------------------------------
    support_idx = [-1]
    val_wsize = 2
    n_sup_part = 3

    # Network ---------------------------------------------------------------
    modelname = 'dinov3_vits16'
    clsname = 'grid_proto'
    proto_grid_size = 8
    feature_hw = [max(input_size[0] // 16, 32), max(input_size[0] // 16, 32)]
    reload_model_path = None
    adapter_state_path = os.environ.get('STAGE1_ADAPTER_PATH')
    lora = int(os.environ.get('STAGE2_LORA_RANK', '8'))
    adapter_channels = 3
    use_slice_adapter = False
    adapter_layers = 1

    support_id_whitelist = _load_id_list(os.environ.get('SUPPORT_TILE_FILE'))
    train_query_id_whitelist = _load_id_list(os.environ.get('STAGE2_TRAIN_QUERY_FILE'))

    model = {
        'align': usealign,
        'which_model': modelname,
        'cls_name': clsname,
        'proto_grid_size' : proto_grid_size,
        'feature_hw': feature_hw,
        'reload_model_path': reload_model_path,
        'adapter_state_path': adapter_state_path,
        'lora': lora,
        'adapter_channels': adapter_channels,
        'use_slice_adapter': use_slice_adapter,
        'adapter_layers': adapter_layers,
        'debug': False,
    }

    task = {
        'n_ways': 1,
        'n_shots': 1,
        'n_queries': 1,
        'npart': n_sup_part 
    }

    optim_type = 'sgd'
    lr=1e-3
    momentum=0.9
    weight_decay=0.0005
    optim = {
        'lr': lr, 
        'momentum': momentum,
        'weight_decay': weight_decay
    }

    dataset_act_labels = {
        'POTSDAM_BIJIE': [1],
        'POTSDAM_OPENEARTHMAP': list(range(1, 9)),
    }

    exp_prefix = ''

    exp_str = '_'.join([exp_prefix, dataset, f'sets_{task["n_shots"]}_shot'])

    path = {
        'log_dir': './runs',
        'POTSDAM_BIJIE': {'data_dir': './data/potsdam_bijie'},
        'POTSDAM_OPENEARTHMAP': {'data_dir': './data/potsdam_OpenEarthMap'},
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
