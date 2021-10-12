import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path

from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid import ConvTasNet
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates
from asteroid.metrics import WERTracker, MockWERTracker

from datetime import datetime
from datetime import date
from asteroid.models import BaseModel
from collections import OrderedDict
from asteroid.dsp.normalization import normalize_estimates, normalize_estimates_cascaded_tensor
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir", type=str, default=r'C:\Users\eliashivk\Documents\DSP_final_project\asteroid-master\egs\librimix\ConvTasNet\local\data\wav8k\min\metadata\test', help="Test directory including the csv files"
)
parser.add_argument(
    "--task",
    type=str,
    default='sep_noisy',
    help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=r'C:\Users\eliashivk\Documents\DSP_final_project\asteroid-master\egs\librimix\ConvTasNet\exp\Test_on_test_free_cascaded_noFineTuning',
    help="Directory in exp_dir where the eval results" " will be stored"
)
parser.add_argument(
    "--use_gpu", type=int, default=1, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default=r'exp\train_free_cascaded', help="Experiment root")

parser.add_argument(
    "--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all"
)
parser.add_argument(
    "--compute_wer", type=int, default=0, help="Compute WER using ESPNet's pretrained model"
)

parser.add_argument(
    "--partial_set", type=int, default=0, help="taking only the first samples of the mixures in test set, 0 -off, >0 -number of samples"
)

parser.add_argument(
    "--auto_dir_name", type=int, default=0, help="whether to name the out dir automatically"
)

parser.add_argument(
    "--fine_tuning", type=int, default=0, help=""
)

parser.add_argument("--enhc_dir", default="exp/train_free_enhc_both", help="path to enhancement model")

parser.add_argument("--sep_dir", default="exp/train_free_sep-clean", help="path to separation model")

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]
ASR_MODEL_PATH = (
    "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
)


def update_compute_metrics(compute_wer, metric_list):
    if not compute_wer:
        return metric_list
    try:
        from espnet2.bin.asr_inference import Speech2Text
        from espnet_model_zoo.downloader import ModelDownloader
    except ModuleNotFoundError:
        import warnings

        warnings.warn("Couldn't find espnet installation. Continuing without.")
        return metric_list
    return metric_list + ["wer"]


def main(conf):
    compute_metrics = update_compute_metrics(conf["compute_wer"], COMPUTE_METRICS)
    # anno_df = pd.read_csv(Path(conf["test_dir"]).parent.parent.parent / "test_annotations.csv")
    anno_df = pd.read_csv(Path(conf["test_dir"]).parent.parent.parent.parent / "test_annotations.csv")
    wer_tracker = (
        MockWERTracker() if not conf["compute_wer"] else WERTracker(ASR_MODEL_PATH, anno_df)
    )

    # upload models:

    # model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
    # model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    # model = ConvTasNet.from_pretrained(model_path)

    # upload each model separatly
    model_enhc_path = os.path.join(conf["enhc_dir"], "best_model.pth")
    model_enhc = ConvTasNet.from_pretrained(model_enhc_path)
    model_sep_path = os.path.join(conf["sep_dir"], "best_model.pth")
    model_sep = ConvTasNet.from_pretrained(model_sep_path)

    if conf["fine_tuning"]:  # if there's fine-tuning update state dict with the last epoch
        if True:
            state_dict_cascaded = torch.load(
                r'C:\Users\eliashivk\Documents\DSP_final_project\asteroid-master\egs\librimix\ConvTasNet\exp\train_free_cascaded\checkpoints\epoch=27-step=194599.ckpt',
                )
        else:
            model_cascaded_path = os.path.join(r'C:\Users\eliashivk\Documents\DSP_final_project\asteroid-master\egs\librimix\ConvTasNet\exp\train_gammatone_cascaded', "best_model.pth")
            model_cascaded = ConvTasNet.from_pretrained(model_cascaded_path)
            state_dict_cascaded = model_cascaded.serialize()

        # state_dict_enhc = OrderedDict()
        # state_dict_sep = OrderedDict()
        state_dict_enhc = model_enhc.serialize()
        state_dict_sep = model_sep.serialize()
        for k, v in state_dict_cascaded['state_dict'].items():
            if 'model_enhc' in k:
                new_k = k.replace('model_enhc.', '')
                state_dict_enhc['state_dict'][new_k] = v
                # print(new_k)
            elif 'model_sep' in k:
                new_k = k.replace('model_sep.', '')
                state_dict_sep['state_dict'][new_k] = v
                # print(new_k)
        model_enhc = ConvTasNet.from_pretrained(state_dict_enhc)
        model_sep = ConvTasNet.from_pretrained(state_dict_sep)


    # Handle device placement
    if conf["use_gpu"]:
        model_enhc.cuda()
        model_sep.cuda()
    model_device = next(model_enhc.parameters()).device
    test_set = LibriMix(
        csv_dir=conf["test_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=conf["train_conf"]["data"]["n_src"],
        segment=None,
        return_id=True,
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    # eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    # shira and karine - 21/07/21
    if conf["auto_dir_name"]==1 :
        eval_save_dir = os.path.join(conf["out_dir"],
                                     'Test_results_{}_{}'.format(date.today().strftime("%d-%m-%Y"),
                                                                 datetime.now().strftime("%H-%M-%S")))
    else:
        eval_save_dir = conf["out_dir"]
    info_mat = {'date': date.today().strftime("%d/%m/%Y"),
                'time': datetime.now().strftime("%H:%M:%S"),
                'test_dir': conf["test_dir"],
                'task': conf["task"],
                'model_dir': conf["exp_dir"]}
    os.makedirs(eval_save_dir, exist_ok=True)
    with open(eval_save_dir + "\\test_info.json", "w") as f:
        json.dump(info_mat, f, indent=0)
    ############
    ex_save_dir = os.path.join(eval_save_dir, "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    # save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    save_idx = range(conf["n_save_ex"]) # karine and shira - 22/7/21
    series_list = []
    torch.no_grad().__enter__()
    if conf["partial_set"] > 0:
        set_len = conf["partial_set"]
    else:
        set_len = len(test_set)
    for idx in tqdm(range(set_len)):
        # Forward the network on the mixture.
        mix, sources, ids = test_set[idx]
        mix, sources = tensors_to_device([mix, sources], device=model_device)

        # forward cascaded:
        mix_after_denoise = model_enhc(mix.unsqueeze(0))
        mix_after_denoise_norm = normalize_estimates_cascaded_tensor(mix_after_denoise, mix.unsqueeze(0))
        # est_sources = model_sep(mix_after_denoise_norm[:, 1, :])  # taking only mixture and discarding the noise
        est_sources = model_sep(mix_after_denoise_norm[:, 0, :])  # taking only mixture and discarding the noise

        # calculate loss
        loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
        mix_np = mix.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()

        # visualize
        if False:
            s1 = sources_np[0, :]
            s2 = sources_np[1, :]
            plt.plot(s1)
            fig, axs = plt.subplots(4, 1)
            axs[0].plot(s1)
            axs[0].set_title('S1')
            axs[1].plot(s2)
            axs[1].set_title('S2')
            s1_fft = np.log10(abs(np.fft.fft(s1)))
            s2_fft = np.log10(abs(np.fft.fft(s2)))
            axs[2].plot(np.arange(s1.size)*(8000/s1.size), np.log10(abs(np.fft.fft(s1))))
            axs[2].set_title('s1 fft')
            axs[3].plot(np.arange(s2.size)*(8000/s2.size), np.log10(abs(np.fft.fft(s2))))
            axs[3].set_title('s2 fft')

            plt.show()

        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["sample_rate"],
            metrics_list=COMPUTE_METRICS,
        )
        utt_metrics["mix_path"] = test_set.mixture_path
        est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)
        utt_metrics.update(
            **wer_tracker(
                mix=mix_np,
                clean=sources_np,
                estimate=est_sources_np_normalized,
                wav_id=ids,
                sample_rate=conf["sample_rate"],
            )
        )
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np, conf["sample_rate"])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx), src, conf["sample_rate"])
            for src_idx, est_src in enumerate(est_sources_np_normalized):
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx),
                    est_src,
                    conf["sample_rate"],
                )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    pprint(final_results)
    if conf["compute_wer"]:
        print("\nWER report")
        wer_card = wer_tracker.final_report_as_markdown()
        print(wer_card)
        # Save the report
        with open(os.path.join(eval_save_dir, "final_wer.md"), "w") as f:
            f.write(wer_card)

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

    model_dict_enhc = torch.load(model_enhc_path, map_location="cpu")
    model_dict_sep = torch.load(model_sep_path, map_location="cpu")
    os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
    publishable = save_publishable(
        os.path.join(conf["exp_dir"], "publish_dir"),
        model_dict_sep,
        metrics=final_results,
        train_conf=train_conf,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )

    main(arg_dic)
