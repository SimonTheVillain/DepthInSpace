import numpy as np
import torch
from pathlib import Path
from model import networks
import h5py
import cv2
import time

#todo: get this from the config file
data_path = str(Path("/media/simon/T7/datasets/DepthInSpace/"))
data_inds_captured = np.arange(4, 147, 8)
data_inds_rendered = np.arange(2 ** 9, 2 ** 10)
thresholds = [0.1, 0.5, 1, 2, 5]

algorithms = [  #("syn_default", "synthetic_default_patt/single_frame/net_0000.params", "rendered_default", data_inds_rendered),
                ("syn_kinect", "synthetic_kinect_patt/single_frame/net_0000.params", "rendered_kinect", data_inds_rendered),
                ("syn_real", "dis_real_lcn_c960_chk.pt", "rendered_real", data_inds_rendered),
                ("real", "dis_real_lcn_j2_c960_chk.pt", "captured", data_inds_captured)
                ]

for name, model_name, subpath, inds in algorithms:
    print("-" * 80)
    print(name)

    print("-" * 80)
    print()

    imsizes = [[512, 432]]  # [settings['imsize']]
    for iter in range(3):
        imsizes.append((int(imsizes[-1][0] / 2), int(imsizes[-1][1] / 2)))
    model = networks.DispDecoder(channels_in=2, max_disp=128, imsizes=imsizes)
    model.load_state_dict(torch.load(f"trained_models/{model_name}"))
    model.cuda()
    lcn = networks.LCN(5, 0.05).cuda()

    data = {}
    for th in thresholds:
        data[f"outliers_{th}"] = {
            "outliers": 0,
            "valid": 0
        }
    for ind in inds:
        #todo: load data
        f = h5py.File(f'{data_path}/{subpath}/{ind:08d}/frames.hdf5', 'r')
        im = torch.tensor(np.array(f["im"])).cuda()
        #todo: do some

        disp = torch.tensor(np.array(f["disp"])).cuda()
        #im = np.array(f["im"])
        #disp = np.array(f["disp"])
        #todo: run_network
        with torch.no_grad():

            tsince = int(round(time.time() * 1000))
            result = model(torch.cat([lcn(im)[0], im], dim=1))
            result = result[0]
            torch.cuda.synchronize()
            ttime_elapsed = int(round(time.time() * 1000)) - tsince
            msk = disp != 0
            valid_count = torch.count_nonzero(msk).item()
            delta = torch.abs(result - disp)

            if False:
                print(f"test time elapsed {ttime_elapsed } ms")
                cv2.imshow("im", im[0,0,:,:].detach().cpu().numpy())
                cv2.imshow("disp", disp[0,0,:,:].detach().cpu().numpy()/10)
                cv2.imshow("result", result[0,0,:,:].detach().cpu().numpy()/10)
                cv2.imshow("delta", delta[0,0,:,:].detach().cpu().numpy()/10)
                cv2.waitKey()

            for th in thresholds:
                outlier_count = torch.count_nonzero(torch.logical_and(delta > th, msk))
                data[f"outliers_{th}"]["outliers"] += outlier_count.item()
                data[f"outliers_{th}"]["valid"] += valid_count

    for th in thresholds:
        outlier_ratio = data[f"outliers_{th}"]["outliers"] / data[f"outliers_{th}"]["valid"]
        print(f"o({th}) = {outlier_ratio}")
