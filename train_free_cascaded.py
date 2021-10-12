import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.models import ConvTasNet
from asteroid.data import LibriMix
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system_cascaded import System  # change for cascaded
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

from collections import OrderedDict

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/train_free_gammatone_cascaded", help="Full path to save best validation model")

parser.add_argument("--enhc_dir", default="exp/train_free_enhc_both", help="path to enhancement model")

parser.add_argument("--sep_dir", default="exp/train_gammatone_sep-clean", help="path to separation model")

parser.add_argument("--is_pre_trained", default=False, help="true = load pre trained model from online, false = load models checkpoints")


def main(conf):
    train_set = LibriMix(
        csv_dir=conf["data"]["train_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
    )

    val_set = LibriMix(
        csv_dir=conf["data"]["valid_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    if conf["main_args"]["is_pre_trained"]:
        # upload pretrained from online:
        model_enhc = ConvTasNet.from_pretrained("mpariente/ConvTasNet_Libri1Mix_enhsingle_8k")
        model_sep = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
    else:
        # model_enhc = ConvTasNet(
        #     **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"]
        # )
        # model_sep = ConvTasNet(
        #     **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"]
        # )

        model_enhc_path = os.path.join(conf["main_args"]["enhc_dir"], "best_model.pth")
        model_enhc = ConvTasNet.from_pretrained(model_enhc_path)
        model_sep_path = os.path.join(conf["main_args"]["sep_dir"], "best_model.pth")
        model_sep = ConvTasNet.from_pretrained(model_sep_path)


    optimizer = make_optimizer(list(model_sep.parameters())+list(model_enhc.parameters()), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model_enhc=model_enhc,
        model_sep=model_sep,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Loading state dicts of pretrained models for enhancement and separation
    state_dict_enhc = model_enhc.state_dict()
    new_state_enhc_dict = OrderedDict()
    for k, v in state_dict_enhc.items():
        if 'model_enhc' not in k:
            k = k.replace('model', 'model_enhc')
            # print(k)
        new_state_enhc_dict[k] = v
    load_state_enhc = system.load_state_dict(state_dict=new_state_enhc_dict, strict=False)

    state_dict_sep = model_sep.state_dict()
    new_state_sep_dict = OrderedDict()
    for k, v in state_dict_sep.items():
        if 'model_sep' not in k:
            k = k.replace('model', 'model_sep')
            # print(k)
        new_state_sep_dict[k] = v
    load_state_sep = system.load_state_dict(state_dict=new_state_sep_dict, strict=False)

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    # distributed_backend = "ddp" if torch.cuda.is_available() else None
    distributed_backend = None

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        resume_from_checkpoint = None  #r'C:\Users\eliashivk\Documents\DSP_final_project\asteroid-master\egs\librimix\ConvTasNet\exp\train_gammatone_cascaded\checkpoints\epoch=27-step=194599.ckpt'   # Karine and Shira - 9/7/21
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    new_state_cascaded_dict = OrderedDict()

    state_dict_enhc = system.model_enhc.serialize()
    for k, v in state_dict_enhc['state_dict'].items():
        new_k = 'model_enhc.' + k
        new_state_cascaded_dict[new_k] = v
    state_dict_sep = system.model_sep.serialize()
    for k, v in state_dict_sep['state_dict'].items():
        new_k = 'model_sep.' + k
        new_state_cascaded_dict[new_k] = v

    state_dict_enhc['state_dict'] = new_state_cascaded_dict
    to_save = state_dict_enhc

    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf_cascaded.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
