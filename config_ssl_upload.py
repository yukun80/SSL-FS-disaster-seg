"""
Experiment configuration file
Extended from config file from original PANet Repository
"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from platform import node
from datetime import datetime

from util.consts import IMG_SIZE

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('mySSL')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations"""
    seed = 1234
    gpu_id = 0
    mode = 'train' # for now only allows 'train' 
    do_validation=False
    num_workers = 4 # 0 for debugging. 

    dataset = 'POTSDAM_BIJIE'  # remote-sensing landslide dataset
    use_coco_init = False  # hub backbones already self-supervised

    ### Training
    n_steps = 100100
    batch_size = 1
    lr_milestones = [ (ii + 1) * 1000 for ii in range(n_steps // 1000 - 1)]
    lr_step_gamma = 0.95
    ignore_label = 255
    print_interval = 100
    save_snapshot_every = 25000
    max_iters_per_load = 1000 # epoch size, interval for reloading the dataset
    epochs=1
    scan_per_load = -1
    which_aug = 'satellite_aug'
    input_size = (512, 512)
    min_fg_data='0'
    label_sets = 0
    curr_cls = ""
    exclude_cls_list = []
    usealign = True # keep prototype alignment loss
    use_wce = True
    use_dinov2_loss = False
    dice_loss = False
    ### Validation
    z_margin = 0 
    eval_fold = 0 # which fold for 5 fold cross validation
    support_idx=[-1] # indicating which scan is used as support in testing. 
    val_wsize=2 # L_H, L_W in testing
    n_sup_part = 3 # number of chuncks in testing
    use_clahe = False
    use_slice_adapter = False
    adapter_layers=3
    debug=False
    skip_no_organ_slices=True
    # Network
    modelname = 'dinov3_vits16'
    encodername='default' # relevant to Mask2Former
    clsname = 'grid_proto'
    reload_model_path = None # path for reloading a trained model (overrides ms-coco initialization)
    proto_grid_size = 8 # L_H, L_W = (32, 32) / 8 = (4, 4)  in training
    feature_hw = [max(input_size[0] // 16, 32), max(input_size[0] // 16, 32)]
    lora = 0
    use_3_slices=False
    do_cca=False
    use_edge_detector=False
    finetune_on_support=False
    sliding_window_confidence_segmentation=False
    finetune_model_on_single_slice=False
    online_finetuning=True
    predict_tp_slices=False
    use_bbox=True # for SAM
    use_points=True # for SAM
    use_mask=False # for SAM
    base_model="alpnet" # or "autosam"
    # SSL
    superpix_scale = 'MIDDLE' #MIDDLE/ LARGE
    use_pos_enc=False
    support_txt_file = None # optional path listing support tile ids (one per line)
    support_id_whitelist = None
    if support_txt_file is not None and os.path.isfile(support_txt_file):
        with open(support_txt_file, 'r', encoding='utf-8') as f:
            support_id_whitelist = [line.strip() for line in f if line.strip()]
    augment_support_set=False
    coarse_pred_only=False # for ProtoSAM 
    point_mode="both" # for ProtoSAM, choose: both, conf, centroid
    use_neg_points=False
    n_support=5 # num support images
    protosam_sam_ver="sam" # or medsam
    protosam_sam_size="huge" # huge or base, med
    grad_accumulation_steps=1
    ttt=False
    reset_after_slice=True # for TTT, if to reset the model after finetuning on each slice
    model = {
        'align': usealign,
        'dinov2_loss': use_dinov2_loss,
        'use_coco_init': use_coco_init,
        'which_model': modelname,
        'cls_name': clsname,
        'proto_grid_size' : proto_grid_size,
        'feature_hw': feature_hw,
        'reload_model_path': reload_model_path,
        'lora': lora,
        'use_slice_adapter': use_slice_adapter,
        'adapter_layers': adapter_layers,
        'debug': debug,
        'use_pos_enc': use_pos_enc
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

    exp_prefix = ''

    exp_str = '_'.join(
        [exp_prefix]
        + [dataset,]
        + [f'sets_{label_sets}_{task["n_shots"]}shot'])

    path = {
        'log_dir': './runs',
        'POTSDAM_BIJIE': {'data_dir': './data/potsdam_bijie'},
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
