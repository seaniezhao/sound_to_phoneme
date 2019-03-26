from model.wavenet_model import WaveNetModel
from model.model_training import ModelTrainer
import hparams
import torch
from data.dataset import STPDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = WaveNetModel(hparams.create_hparams(), device).to(device)
print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())
data = STPDataset(data_folder='data/prepared_data', receptive_field=model.receptive_field,)
print('the dataset has ' + str(len(data)) + ' items')

trainer = ModelTrainer(model=model,
                         dataset=data,
                         lr=0.0005,
                         weight_decay=0.0,
                         snapshot_path='./snapshots',
                         snapshot_name='stp_model',
                         snapshot_interval=2000,
                         device=device)


print('start training...')
trainer.train(batch_size=32,
              epochs=50)
