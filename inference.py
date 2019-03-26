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


model = load_latest_model_from('snapshots')

data = np.load('data/prepared_data/data.npy')
phn_list = model.get_phonetic(data)
for i in phn_list:
    print(i.numpy())

