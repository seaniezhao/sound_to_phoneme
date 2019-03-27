from model.wavenet_model import *
import hparams
import matplotlib.pyplot as plt
import numpy as np


def load_latest_model_from(location):

    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)

    print("load model " + newest_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hparam = hparams.create_hparams()

    model = WaveNetModel(hparam, device).to(device)
    states = torch.load(newest_file)
    model.load_state_dict(states['state_dict'])
    return model


# 1 frame equals 8ms
def get_time(phn_list, all_phn):

    phn_timing = []

    current_idx = -1
    current_start = 0
    for idx, item in enumerate(phn_list):
        if current_idx == -1:
            current_start = idx * 8
            current_idx = item
        elif current_idx != item or idx == len(phn_list)-1:
            end = idx*8
            phn_timing.append((current_start, end, all_phn[current_idx]))
            current_start = end
            current_idx = item

    return phn_timing



all_phn = np.load('data/all_phn.npy')
model = load_latest_model_from('snapshots')

data = np.load('data/prepared_data/000001_data.npy')
phn_list = model.get_phonetic(data)

phn_timing = get_time(phn_list, all_phn)
for i in phn_timing:
    print(i)

