import matplotlib
matplotlib.use('Agg')
import torch
import cv2
import numpy as np

#Work in the parent directory
import os
import model
from model.networks import *
import torch
import os
import cv2
import numpy as np
import re
from model import networks
from pathlib import Path

mode = "rendered" # vs captured
architecture = "single_frame"
half_res = True
focal = 1
baseline = 1
baseline_correction = 0.0634 / 0.07501
baseline_correction = 1.0

if mode == "rendered":
    src_res = (1401, 1001)
    src_cxy = (700, 500)
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03

    rrl = (src_cxy[0] - tgt_cxy[0], src_cxy[1] - tgt_cxy[1], tgt_res[0], tgt_res[1])
    path = "/media/simon/T7/datasets/structure_core_unity_test"
    path_out = "/media/simon/T7/datasets/structure_core_unity_test_results/DepthInSpaceSF"
    state_dict = torch.load("output_structure_unity_stereo/exp_syn/net_1429.params")

    path_out = "/media/simon/ssd_datasets/datasets/structure_core_unity_test_results/connecting_the_dots_full"
    state_dict = torch.load("output_structure_unity_direct/exp_syn/net_1269.params")

    inds = os.listdir(path)
    inds = [re.search(r'\d+', s).group() for s in inds]
    inds = set(inds)
    inds = list(inds)
    inds.sort()
    paths = []
    for ind in inds:
        tgt_pth = Path(path_out) / f"{ind}.exr"
        paths.append((path + f"/{ind}_left.jpg", tgt_pth))
elif mode=="captured":
    tgt_res = (1216, 896)
    tgt_cxy = (604, 457)
    # the focal length is shared between src and target frame
    focal = 1.1154399414062500e+03

    rrl = (tgt_res[0], 0, tgt_res[0], tgt_res[1])
    path = "/media/simon/ssd_datasets/datasets/structure_core_photoneo_test"
    path_out = "/media/simon/ssd_datasets/datasets/structure_core_photoneo_test_results/connecting_the_dots"

    path = "/media/simon/ssd_datasets/datasets/structure_core_photoneo_test"
    path_out = "/media/simon/ssd_datasets/datasets/structure_core_photoneo_test_results/connecting_the_dots"
    state_dict = torch.load("output_structure_captured_stereo/exp_syn/net_2779.params")
    folders = os.listdir(path)
    scenes = [x for x in folders if os.path.isdir(Path(path) / x)]

    paths = []
    for scene in scenes:
        tgt_path = Path(path_out) / scene
        if not os.path.exists(tgt_path):
            os.mkdir(tgt_path)
        for i in range(4):
            pl = Path(path) / scene / f"ir{i}.png"
            tgt_pth = Path(tgt_path) / f"{i}.exr"
            paths.append((str(pl), str(tgt_pth)))

# set up network
if architecture == "single_frame":
    imsizes = [(896//2, 1216//2)]
    for iter in range(3):
        imsizes.append((int(imsizes[-1][0]/2), int(imsizes[-1][1]/2)))
    net = networks.DispDecoder(channels_in=2, max_disp=128, imsizes=imsizes)
    net.load_state_dict(state_dict)
    net.cuda()

    lcn = LCN(5, 0.05)
    lcn.cuda()


with torch.no_grad():
    for psrc, pout in paths:
        p = psrc
        irl = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if len(irl.shape) == 3:
            # the rendered images are 3 channel bgr
            irl = cv2.cvtColor(irl, cv2.COLOR_BGR2GRAY)
        else:
            # the rendered images are 16
            irl = irl / 255.0
        irl = irl[rrl[1]:rrl[1] + rrl[3], rrl[0]:rrl[0] + rrl[2]]
        irl = irl.astype(np.float32) * (1.0 / 255.0)


        if half_res:
            irl = cv2.resize(irl, (int(irl.shape[1] / 2), int(irl.shape[0] / 2)))

        irl = irl[:448, :608]
        cv2.imshow("irleft", irl)
        irl = torch.tensor(irl).cuda().unsqueeze(0).unsqueeze(0)

        #run local contrast normalization (LCN)

        lc, _ = lcn(irl)
        irl = torch.cat((lc, irl), 1)
        disp, edge = net(irl)

        result = disp[0].cpu()[0, 0, :, :].numpy()
        #result = coresup_pred.cpu()[0, 0, :, :].numpy()

        p = str(pout)#path_out + f"/{int(ind):05d}.exr"
        cv2.imshow("result", result * (1.0 / 50.0))
        cv2.imwrite(p, result * baseline_correction)
        cv2.waitKey(1)
        print(p)