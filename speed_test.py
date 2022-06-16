import torch
from model import single_frame_worker
from model import multi_frame_worker
from model import networks
from model import multi_frame_networks
from co.args import parse_args
import time

net_path = "trained_models/synthetic_real_patt/single_frame/net_0000.params"
state_dict = torch.load(str(net_path))

imsizes = [[512, 432]]#[settings['imsize']]
for iter in range(3):
    imsizes.append((int(imsizes[-1][0] / 2), int(imsizes[-1][1] / 2)))

net = networks.DispDecoder(channels_in=2, max_disp=128, imsizes=imsizes)
net.load_state_dict(state_dict)
net.cuda()

test_data = torch.zeros((1, 2, 448, 604)).cuda() #  512, 432/ 448,604
runs = 10
with torch.no_grad():
    torch.cuda.synchronize()
    _ = net(test_data)
    torch.cuda.synchronize()
    tsince = int(round(time.time() * 1000))
    for i in range(0, runs):
        _ = net(test_data)
        torch.cuda.synchronize()
    ttime_elapsed = int(round(time.time() * 1000)) - tsince
    print(f"test time elapsed {ttime_elapsed / runs} ms")