import torch
import cv2
import random
import os.path as osp
import pandas as pd
from model import DiViDeAddEvaluator
from datasets import FusionDataset

import argparse

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

from time import time
from tqdm import tqdm
import pickle
import math

import yaml
import csv
from thop import profile



official_score_distribution_dict = {
    1.1: 180, 1.3: 360, 1.5: 375, 1.7: 160, 1.9:  99,
    2.1:  80, 2.3:  79, 2.5:  98, 2.7: 155, 2.9: 245,
    3.1: 330, 3.3: 395, 3.5: 390, 3.7: 365, 3.9: 475,
    4.1: 687, 4.3: 800, 4.5: 500, 4.7: 130, 4.9:  10,
}

def official_collect_dist():
    values = []
    for score, count in official_score_distribution_dict.items():
        values.extend([score] * count)
    return values

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments"]


def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device)
            c, t, h, w = video[key].shape
            video[key] = video[key].reshape(1, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape( data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
    with torch.no_grad():
        flops, params = profile(model, (video, ))
    print(f"The FLOps of the Variant is {flops/1e9:.1f}G, with Params {params/1e6:.2f}M.")

def inference_set(inf_loader, model, device, output_file, args, save_model=False, set_name="na"):
    print(f"Validating for {set_name}.")
    results = []
    video_paths = []
    keys = []
    anno_file = args["anno_file"]
    load_csv = args["load_csv"]
    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video = {}
        for key in sample_types:
            if key not in keys:
                keys.append(key)
            if key in data:
                video[key] = data[key].to(device)
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
        with torch.no_grad():
            labels = model(video,reduce_scores=False)
            labels = [np.mean(l.cpu().numpy()) for l in labels]
            result["pr_labels"] = labels
        video_path = data["name"][0]
        video_paths.append(video_path) 
        result["gt_label"] = data["gt_label"].item()
        result["name"] = data["name"]
        results.append(result)
    
    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = 0
    pr_dict = {}
    for i, key in zip(range(len(results[0]["pr_labels"])), keys):
        key_pr_labels = np.array([np.mean(r["pr_labels"][i]) for r in results])
        pr_dict[key] = key_pr_labels
        pr_labels += rescale(key_pr_labels)
    pr_labels = rescale(pr_labels, gt_labels if anno_file else official_collect_dist()) #resize pr_labels to the same scale as gt_labels

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())
    print(
        f"For {len(inf_loader)} videos, \nthe accuracy of the model is as follows:\n  SROCC: {s:.4f} \n  PLCC:  {p:.4f} \n  KROCC: {k:.4f} \n  RMSE:  {r:.4f}."
    )

    if not anno_file and load_csv:
        base_df = pd.read_csv(load_csv)
        new_df = pd.DataFrame({"video":video_paths, "koala_score":np.round(pr_labels,3)})
        merged = base_df.merge(new_df, on="video", how="left")
        merged.to_csv(output_file, index=False)

    else:
        pd.DataFrame({"video":video_paths, 
                      "koala_score":np.round(pr_labels,3)}).to_csv(output_file, index=False) 


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="test.yml", help="the option file"
    )
    parser.add_argument(
        "-t", "--output_csv", type=str, default=f"../../koala_score_video.csv", help="the output file"
    )
    parser.add_argument("--data_prefix", type=str, default="./", help="video data path")
    parser.add_argument("--load_csv", type=str, default="", help="load video annotation csv file")
    args = parser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
        opt["data"]["test-data"]["args"]['data_prefix'] = args.data_prefix
        opt["data"]["test-data"]["args"]['load_csv'] = args.load_csv
        

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)

    # state_dict = torch.load(opt["test_load_path"], map_location=device)["state_dict"]
    state_dict = torch.load(opt["test_load_path"], map_location=device, weights_only=False)["state_dict"]

    
    if "test_load_path_aux" in opt:
        aux_state_dict = torch.load(opt["test_load_path_aux"], map_location=device)["state_dict"]
        
        from collections import OrderedDict
        
        fusion_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "fragments")
            else:
                ki = k
            fusion_state_dict[ki] = v
            
        for k, v in aux_state_dict.items():
            if k.startswith("frag"):
                continue
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "resize")
            else:
                ki = k
            fusion_state_dict[ki] = v
        
        state_dict = fusion_state_dict
        
    model.load_state_dict(state_dict, strict=True)
    for key in opt["data"].keys(): # different datasets
        
        if "val" not in key and "test" not in key:
            continue
        
        val_dataset = FusionDataset(opt["data"][key]["args"])


        val_loader =  torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )

        inference_set(
            val_loader,
            model,
            device, 
            args.output_csv,
            opt["data"]["test-data"]["args"],
            set_name=key,

        )



if __name__ == "__main__":
    main()
