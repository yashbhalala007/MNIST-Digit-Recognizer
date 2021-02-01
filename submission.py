import torch
from train import NNet, CNNet
import numpy as np
import pandas as pd

def create_file(filename, pred):
	subm = []
	for i in range(len(pred)):
	  subm.append([i+1, np.argmax(pred[i].cpu().detach().numpy())])
	subm_file = pd.DataFrame(subm, columns=['ImageId', 'Label'])
	subm_file.to_csv(filename, index=False)

fc_model = torch.load("fc_net.pth", map_location=torch.device('cpu'))
fc_model.eval()

cnn_model = torch.load("cnn_net.pth", map_location=torch.device('cpu'))
cnn_model.eval()

test = pd.read_csv('test.csv')
df_test = torch.from_numpy(test.values).float() / 255

fc_pred = fc_model(df_test)
cnn_pred = cnn_model(df_test.view(-1, 1, 28, 28))

create_file('fc_submission.csv', fc_pred)
create_file('cnn_submission.csv', cnn_pred)