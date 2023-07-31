from typing import Any
import pickle
from alpha_net_c4 import ConnectNet, board_data
from argparse import ArgumentParser
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import sys
import logging
import numpy as np

class LinearProbes():
    def __init__(self, model_file):
        self.layer_activations = []
        self.model = ConnectNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load a saved model. Note that the architecture must match that of self.model.
        try:
            checkpoint = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])

            # Set model to evaluation mode.
            self.model.eval()

        except Exception as e:
            print(f'An err occurred: {e}')
            sys.exit(1)

    def register_activations(self):
        # Get list of layers except for the final outblock.
        layers = list(self.model.children())[:-1]
        print("num layers: ", len(layers))
        
        self.layer_activations = [None]*len(layers)

        for i, layer in enumerate(layers):
            layer.register_forward_hook(self.get_activations_hook(i))


    def get_activations_hook(self, name):
        def hook(model, input, output):
            self.layer_activations[name] = output.detach()
        return hook

    def load_data(self, dataset_file):
        with open(dataset_file, 'rb') as pkl_file:
            data = pickle.load(pkl_file, encoding='bytes')
            data = np.array(data)
            data = board_data(data)
        return data

    def generate_dataset(self, data):
        self.register_activations()

        board_states = []
        for d in data:
            board_states.append(d[0])
        board_states = torch.tensor(np.array(board_states), dtype=torch.float)

        policy, value = self.model(board_states) 

        activations = [None]*len(self.layer_activations)

        # Flatten activations into a single row for each board_state.
        for i, layer in enumerate(self.layer_activations):
            activations[i] = layer.flatten(start_dim=1)

        return board_states, activations

    def train_linear_models(self, targets, board_states):
        activations = self.generate_dataset(board_states)

        linear_models = [[LogisticRegression() for i in range(len(activations))] for y in targets]
        loss = [[0 for i in range(len(activations))] for y in targets]

        # Train a model for each pair of (target, layer) and record the loss.
        for i, target in enumerate(targets):
            for j, model in enumerate(linear_models[i]):
                model.fit(activations[j], target)
                pred = model.predict_proba(activations[j])
                l = log_loss(target, pred)
                loss[i][j] = l

        return linear_models, loss

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, default="", help="The model file to load into the network.")
    parser.add_argument("--board_data", type=str, default="", help="The file containing the labeled board states.")
    args = parser.parse_args()
    args_dict = vars(args)

    # Log arguments
    arg_str = ""
    for arg_name, arg_value in args_dict.items():
        arg_str = arg_str + f'{arg_name}: {arg_value}\n'
    logger.info("Running linear probes with args: \n" + arg_str)


    data_file = "./datasets/iter_0/dataset_iter0_cpu0_0_2023-07-26"
    model_file = "./model_data/cc4_current_net__iter0.pth.tar"
    lp = LinearProbes(model_file)

    data = lp.load_data(data_file)
    board_states, activations = lp.generate_dataset(data)
    
    print("len(board_states): ", len(board_states))
    print(len(activations[1][1]))
    print(activations[1][1][0])

    a = activations[9]
    print("a.shape: ", a.shape)

    flattened_activations = a.flatten(start_dim=1)
    print("flattened size, shape: ", len(flattened_activations), flattened_activations.shape)


