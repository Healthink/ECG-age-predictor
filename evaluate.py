# Imports
import argparse
import json
import os
from warnings import warn

import numpy as np
import pandas as pd
import torch
import wfdb
from tqdm import tqdm

from branch_nn import Branch_NN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mdl', type=str, default='model',
        help='folder containing model.')
    parser.add_argument('--path_to_traces', type=str, default='../data/ecg_tracings.hdf5',
        help='path to hdf5 containing ECG traces')
    parser.add_argument('--batch_size', type=int, default=8,
        help='number of exams per batch.')
    parser.add_argument('--output', type=str, default='predicted_age.csv',
        help='output file.')
    parser.add_argument('--traces_dset', default='tracings',
        help='traces dataset in the hdf5 file.')
    parser.add_argument('--ids_dset', default=None,
        help='ids dataset in the hdf5 file.')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Get checkpoint
    ckpt = torch.load(os.path.join(args.mdl, 'model.pth'), map_location=lambda storage, loc: storage)
    # Get config
    config = os.path.join(args.mdl, 'args.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)
    # Get model
    N_LEADS = 12
    model = Branch_NN(
        dnn_input_dim=len(config_dict['feats_index']),
        dnn_output_dim=config_dict['dnn_output_dim'],
        cnn_input_dim=(N_LEADS, config_dict['seq_length']),
        blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
        n_classes=1,
        kernel_size=config_dict['kernel_size'],
        dropout_rate=config_dict['dropout_rate']
    )
    # model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
    #     blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
    #     n_classes=1,
    #     kernel_size=config_dict['kernel_size'],
    #     dropout_rate=config_dict['dropout_rate'])

    # load model checkpoint
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    # Get traces
    # ff = h5py.File(args.path_to_traces, 'r')
    # traces = ff[args.traces_dset]


    # get ECG tensors for CNN
    df = pd.read_csv(config_dict['path_to_csv'])
    ages = df[config_dict['age_col']]
    path = config_dict['path_to_traces']
    data2 = []
    for i in tqdm(range(df.shape[0]), desc='loading data: '):
        file = df.iloc[i][3].replace('.ECG', '')
        signal, _ = wfdb.rdsamp(path + '/' + file)
        data2.append(signal)
    traces = np.array(data2)
    traces = traces[:, 0:4096, :]
    print(traces.shape)

    n_total = len(traces)
    ids = range(n_total)

    # get feature tensors for DNN

    feats = df.loc[:, config_dict['feats_index']]
    dnn_feats = feats.fillna(0).values

    print(dnn_feats.shape)

    # Evaluate on test data
    model.eval()
    n_total, n_samples, n_leads = traces.shape
    n_batches = int(np.ceil(n_total / args.batch_size))
    # Compute gradients
    predicted_age = np.zeros((n_total,))

    end = 0
    for i in tqdm(range(n_batches), desc='predicting: '):
        start = end
        end = min((i + 1) * args.batch_size, n_total)
        with torch.no_grad():
            x_cnn = torch.tensor(traces[start:end, :, :]).transpose(-1, -2)
            x_cnn = x_cnn.to(device, dtype=torch.float32)
            x_dnn = torch.tensor(dnn_feats[start:end, :]).to(device, dtype=torch.float32)
            y_pred = model(x_cnn, x_dnn)

        predicted_age[start:end] = y_pred.detach().cpu().numpy().flatten()
    # Save predictions
    df = pd.DataFrame({'ids': ids, 'true_ages': ages, 'predicted_age': predicted_age})
    df = df.set_index('ids')
    df.to_csv(args.output)
