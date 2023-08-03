from typing import Any
import pickle
from alpha_net_c4 import ConnectNet, board_data
from argparse import ArgumentParser
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

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

            board_states = []
            for d in data:
                board_states.append(d[0])

            board_states = np.array(board_states)

        return board_states

    def generate_activations_dataset(self, board_states):
        self.register_activations()

        board_states = torch.tensor(board_states, dtype=torch.float)
        policy, value = self.model(board_states) 
        activations = [None]*len(self.layer_activations)

        # Flatten activations into a single row for each board_state.
        for i, layer in enumerate(self.layer_activations):
            activations[i] = layer.flatten(start_dim=1)
            activations[i] = np.array(activations[i])

        activations = np.array(activations)

        return board_states, activations

    def train_linear_models(self, targets, activations, board_states):
        # Split data into training and test.
        sep = round(len(activations[0])/2)
        training_activations = activations[:, :sep]
        training_labels = targets[:, :sep] 
        test_activations = activations[:, sep:]
        test_labels = targets[:, sep:] 

        linear_models = [[LogisticRegression(max_iter=300) for i in range(len(training_activations))] for y in targets]
        loss = [[0 for i in range(len(test_activations))] for y in targets]
        acc = [[0 for i in range(len(test_activations))] for y in targets]
        f1 = [[0 for i in range(len(test_activations))] for y in targets]

        # Train a model for each pair of (target, layer) and record the loss.
        for i, target in enumerate(training_labels):
            for j, model in enumerate(linear_models[i]):
                model.fit(training_activations[j], target)

                pred_prob = model.predict_proba(test_activations[j])
                pred = model.predict(test_activations[j])

                l = log_loss(test_labels[i], pred_prob)
                a = accuracy_score(test_labels[i], pred)
                f = f1_score(test_labels[i], pred)
                loss[i][j] = l
                acc[i][j] = a
                f1[i][j] = f

        # Train model on raw board data

        return linear_models, loss, acc, f1

# The current player can win the game.
def get_attack_concept_label(boards, concept_fn):
    flag = 0 
    labels = [0]*len(boards)
    for i, board in enumerate(boards):
        labels[i] = concept_fn(board, board[2][0][0])
        #if labels[i] == 1 and flag < 10:
        #    flag += 1
        #    print("attack: \n", board)
    return np.array(labels)

# The player to play next has a game-ending threat.
def get_threat_concept_label(boards, concept_fn):
    flag = 0 
    labels = [0]*len(boards)
    for i, board in enumerate(boards):
        labels[i] = concept_fn(board, [0, 1][board[2][0][0]-1])
        #if labels[i] == 1 and flag < 10:
        #    flag += 1 
        #    print("threat: \n", board)
    return np.array(labels)

def player_has_double_attack(boards):
    flag = 0
    labels = [0]*len(boards)
    for i, board in enumerate(boards):
        labels[i] = has_double_attack(board, board[2][0][0])
        #if labels[i] == 1 and flag < 10:
        #    flag += 1 
        #    print("threat: \n", board)
    return np.array(labels)

def opponent_has_double_attack(boards):
    flag = 0
    labels = [0]*len(boards)
    for i, board in enumerate(boards):
        labels[i] = has_double_attack(board, [0, 1][board[2][0][0]-1])
        #if labels[i] == 1 and flag < 10:
        #    flag += 1 
        #    print("threat: \n", board)
    return np.array(labels)

# Helper function that returns indices of valid moves.
def available_moves(board):
    moves = []
    for col in range(7):
        for row in range(5, -1, -1):
            if board[0, row, col] == 0 and board[1, row, col] == 0:
                moves.append((row, col))
                break
    return moves

# Helper function to check for four consecutive 1's in any direction
def has_four_consecutive_ones(mat):
    rows = len(mat)
    cols = len(mat[0])

    for row in mat:
        for col in range(cols - 3):
            if row[col] == row[col + 1] == row[col + 2] == row[col + 3] == 1:
                return True

    for col in range(cols):
        for row in range(rows - 3):
            if mat[row][col] == mat[row + 1][col] == mat[row + 2][col] == mat[row + 3][col] == 1:
                return True

    for row in range(rows - 3):
        for col in range(cols - 3):
            if mat[row][col] == mat[row + 1][col + 1] == mat[row + 2][col + 2] == mat[row + 3][col + 3] == 1:
                return True

    for row in range(3, rows):
        for col in range(cols - 3):
            if mat[row][col] == mat[row - 1][col + 1] == mat[row - 2][col + 2] == mat[row - 3][col + 3] == 1:
                return True

    return False

# Game ending threat/attack concept.
def can_make_four_consecutive_ones(board, player_number):
    rows = len(board[0])
    cols = len(board[0][0])

    # Check if the matrix already has four consecutive 1's, game already over.
    if has_four_consecutive_ones(board[player_number]):
        return 0
    
    player = board[player_number]

    # If no four consecutive 1's are found, check for the condition with an added '1' for valid moves.
    moves = available_moves(board)
    
    for move in moves:
        row, col = move
        player[row][col] = 1
        if has_four_consecutive_ones(player):
            player[row][col] = 0
            return 1
        player[row][col] = 0
    
    return 0

def has_double_attack(board, player_number):
    mat = board[player_number]
    opponent = board[[0, 1][player_number-1]]
    
    rows = len(mat)
    cols = len(mat[0])

    # Three in a row with open space on both ends.
    for r, row in enumerate(mat):
        for col in range(3):
            if (row[col+1] == row[col + 2] == row[col + 3] == 1) and (opponent[r][col] == opponent[r][col+4] == 0):
                return 1

    # Three in a diagonal (down to right) with open space on both ends.
    for row in range(rows - 4):
        for col in range(cols - 4):
            if (mat[row + 1][col + 1] == mat[row + 2][col + 2] == mat[row+3][col+3] == 1) and (opponent[row][col] == opponent[row+4][col+4] == 0):
                return 1
    
    # Three in other diagonal (up to right) with open space on both ends.
    for row in range(3, rows-1):
        for col in range(cols - 4):
            if (mat[row][col + 1] == mat[row - 1][col + 2] == mat[row - 2][col + 3] == 1) and (opponent[row+1][col] == opponent[row-3][col+4] ==0):
                return 1

    return 0



def run(args):
    logger.info("INVOKED RUN")

    log_args(args)

    board_data = args.board_data
    #board_data = "simple"
    model_name = args.model_name
    #model_name = "vm_training_v3__iter10"

    print("board_data: ", board_data, " model_name: ", model_name)

    model_file = f'./v3_model_data/{model_name}.pth.tar'
    lp = LinearProbes(model_file)

    logger.info("loading data...")
    board_states = load_data(board_data)
    logger.info(f'loaded {len(board_states)} board states')
    logger.info("tagging data...")
    labels, _ = label_data(board_states)
    logger.info("generating activations...")
    _, activations = lp.generate_activations_dataset(board_states)
    #activations = np.array(activations)

    logger.info("training linear models...")
    linear_models, loss = lp.train_linear_models(labels, activations, board_states)
    print("\nATTACK_LOSS: \n", loss[0])
    print("\nTHREAT_LOSS: ", loss[1])

    logger.info("saving loss files")
    loss_file_name = f'./linear_probe_loss/concept_loss-data_{"-".join(board_data)}-model_{model_name}.pkl'
    with open(loss_file_name, 'wb') as file:
        pickle.dump(loss, file)


    logger.info("plotting results...")
    fig, ax = plt.subplots()
    plt.title("Linear Separability of Attack Concept", pad=20)
    plt.xlabel("Layer")
    plt.ylabel("Loss")
    ax.set_xticks(range(len(loss[0])))
    plt.plot(loss[0])
    plt.savefig(f'./linear_probe_loss_plots/attack_concept_loss_layer-data_{"-".join(board_data)}-model_{model_name}.png')
    plt.clf()

    fig, ax = plt.subplots()
    plt.title("Linear Separability of Threat Concept", pad=20)
    plt.xlabel("Layer")
    plt.ylabel("Loss")
    ax.set_xticks(range(len(loss[1])))
    plt.plot(loss[1])
    plt.savefig(f'./linear_probe_loss_plots/threat_concept_loss_layer-data_{"-".join(board_data)}-model_{model_name}.png')
    plt.clf()

################################
#
# Run full experiment suite
#
################################
def run_all(args):
    logger.info("INVOKED RUN_ALL")
    run_all_dir = "./run_all_results"
    #run_all_dir = "./short_iter_results"

    # Get list of all model checkpoints.
    model_files = []
    for i in range(0, 13, 2):
        model_files.append(f'./v3_model_data/vm_training_v3__iter{i}.pth.tar')

    #model_files = []
    #for i in range(0, 15):
    #    model_files.append(f'./model_data/fast_iter_training_v1__iter{i}.pth.tar')

    board_data_dirs = ['iter_0', 'iter_5']

    # Overwrite for testing.
    #model_files = model_files[:3]
    #board_data_dirs = ['subset'] 

    # Load board states, shuffle and add concept labels.
    logger.info("loading data...")
    board_states = load_data(board_data_dirs)
    np.random.shuffle(board_states)
    logger.info(f'loaded {len(board_states)} board states')
    logger.info("tagging data...")
    labels, board_states = label_data(board_states, balance=True)
    
    # For each model train a linear probe for the activations of each layer and save losses and accuracy (recall? precision?).
    losses_by_model = [None]*len(model_files)
    accuracy_by_model = [None]*len(model_files)
    f1_by_model = [None]*len(model_files)
    for idx, model_file in enumerate(model_files):
        logger.info(f'generating activations for model: {idx}...')
        lp = LinearProbes(model_file)
        _, activations = lp.generate_activations_dataset(board_states)
        activations = np.array(activations)

        logger.info(f'training linear models for model: {idx}...')
        linear_models, loss, accuracy, f1 = lp.train_linear_models(labels, activations, board_states)
        losses_by_model[idx] = loss
        accuracy_by_model[idx] = accuracy
        f1_by_model[idx] = f1

    # Get linear model performance on raw board data
    sep = round(len(labels[0])/2)
    training_boards = board_states[:sep]
    test_boards = board_states[sep:]
    training_boards = training_boards.reshape(training_boards.shape[0], -1)
    test_boards = test_boards.reshape(test_boards.shape[0], -1)
    training_labels = labels[:, :sep] 
    test_labels = labels[:, sep:] 

    raw_loss = [[0] for i in labels]
    raw_acc = [[0] for i in labels]
    raw_f1 = [[0] for i in labels]
    for i, target in enumerate(training_labels):
        # Train model on raw board state
        m = LogisticRegression(max_iter=300)
        m.fit(training_boards, target)
        pred_prob = m.predict_proba(test_boards)
        pred = m.predict(test_boards)

        l = log_loss(test_labels[i], pred_prob)
        a = accuracy_score(test_labels[i], pred)
        f = f1_score(test_labels[i], pred)
        raw_loss[i] = l
        raw_acc[i] = a
        raw_f1[i] = f

    print("Performance on raw data:")
    print("loss: ", raw_loss)
    print("acc: ", raw_acc)
    print("f1: ", raw_f1)

    losses_by_model = np.array(losses_by_model)
    accuracy_by_model = np.array(accuracy_by_model)
    f1_by_model = np.array(f1_by_model)

    # Save all losses as pickle
    logger.info("saving run_all loss data...")
    loss_file_name = f'{run_all_dir}/concept_loss-data_run_all.pkl'
    with open(loss_file_name, 'wb') as file:
        pickle.dump(losses_by_model, file)

    accuracy_file_name = f'{run_all_dir}/concept_accuracy-data_run_all.pkl'
    with open(accuracy_file_name, 'wb') as file:
        pickle.dump(accuracy_by_model, file)

    f1_file_name = f'{run_all_dir}/concept_f1-data_run_all.pkl'
    with open(f1_file_name, 'wb') as file:
        pickle.dump(f1_by_model, file)

    # Generate plots:
    def generate_plot(title, xlabel, ylabel, xticks, data, file_name):
        fig, ax = plt.subplots()
        plt.title(title, pad=20)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_xticks(xticks)
        plt.plot(data)
        plt.savefig(file_name)
        plt.clf()


    # For each layer, plot its loss over each iteration.
    concept_name="Double Attack"
    loss_by_layer = np.transpose(losses_by_model, (2, 1, 0))
    accuracy_by_layer = np.transpose(accuracy_by_model, (2, 1, 0))
    f1_by_layer = np.transpose(f1_by_model, (2, 1, 0))
    print("loss_by_layer.shape: ", loss_by_layer.shape)
    for idx in range(len(loss_by_layer)):
        layer_loss = loss_by_layer[idx]
        generate_plot(
            f"Linear Probe Loss for Layer {idx} Activations: {concept_name} Concept", "Iteration", "Loss", range(loss_by_layer.shape[2]),
            layer_loss[0], f'{run_all_dir}/performance_by_layer/attack_concept_layer-{idx}-loss.png')
        #generate_plot(
        #    f"Linear Probe Loss for Layer {idx} Activations: Threat Concept", "Iteration", "Loss", range(loss_by_layer.shape[2]),
        #    layer_loss[1], f'{run_all_dir}/performance_by_layer/threat_concept_layer-{idx}-loss.png')

        layer_acc = accuracy_by_layer[idx]
        generate_plot(
            f"Linear Probe Accuracy for Layer {idx} Activations: {concept_name} Concept", "Iteration", "Accuracy", range(accuracy_by_layer.shape[2]),
            layer_acc[0], f'{run_all_dir}/performance_by_layer/attack_concept_layer-{idx}-accuracy.png')
        #generate_plot(
        #    f"Linear Probe Accuracy for Layer {idx} Activations: Threat Concept", "Iteration", "Accuracy", range(accuracy_by_layer.shape[2]),
        #    layer_acc[1], f'{run_all_dir}/performance_by_layer/threat_concept_layer-{idx}-accuracy.png')

        layer_f1 = f1_by_layer[idx]
        generate_plot(
            f"Linear Probe F1 Score for Layer {idx} Activations: {concept_name} Concept", "Iteration", "F1", range(f1_by_layer.shape[2]),
            layer_f1[0], f'{run_all_dir}/performance_by_layer/attack_concept_layer-{idx}-f1.png')
        #generate_plot(
        #    f"Linear Probe F1 Score for Layer {idx} Activations: Threat Concept", "Iteration", "F1", range(f1_by_layer.shape[2]),
        #    layer_f1[1], f'{run_all_dir}/performance_by_layer/threat_concept_layer-{idx}-f1.png')

    for idx in range(len(losses_by_model)):
        model_iter = idx*2
        model_loss = losses_by_model[idx]
        generate_plot(
            f"Linear Probe Loss for Model Iteration {model_iter} Activations: {concept_name} Concept", "Layer", "Loss", range(losses_by_model.shape[2]),
            model_loss[0], f'{run_all_dir}/performance_by_model/attack_concept_model-{idx}-loss.png')
        #generate_plot(
        #    f"Linear Probe Loss for Model Iteration {idx} Activations: Threat Concept", "Layer", "Loss", range(losses_by_model.shape[2]),
        #    model_loss[1], f'{run_all_dir}/performance_by_model/threat_concept_model-{idx}-loss.png')

        model_acc = accuracy_by_model[idx]
        generate_plot(
            f"Linear Probe Accuracy for Model Iteration {model_iter} Activations: {concept_name} Concept", "Layer", "Accuracy", range(accuracy_by_model.shape[2]),
            model_acc[0], f'{run_all_dir}/performance_by_model/attack_concept_model-{idx}-accuracy.png')
        #generate_plot(
        #    f"Linear Probe Accuracy for Model Iteration {idx} Activations: Threat Concept", "Layer", "Accuracy", range(accuracy_by_model.shape[2]),
        #    model_acc[1], f'{run_all_dir}/performance_by_model/threat_concept_model-{idx}-accuracy.png')

        model_f1 = f1_by_model[idx]
        generate_plot(
            f"Linear Probe F1 Score for Model Iteration {model_iter} Activations: {concept_name} Concept", "Layer", "F1", range(f1_by_model.shape[2]),
            model_f1[0], f'{run_all_dir}/performance_by_model/attack_concept_model-{idx}-f1.png')
        #generate_plot(
        #    f"Linear Probe F1 Score for Model Iteration {idx} Activations: Threat Concept", "Layer", "F1", range(f1_by_model.shape[2]),
        #    model_f1[1], f'{run_all_dir}/performance_by_model/threat_concept_model-{idx}-f1.png')


def log_args(args):
    args_dict = vars(args)
    arg_str = ""
    for arg_name, arg_value in args_dict.items():
        arg_str = arg_str + f'{arg_name}: {arg_value}\n'
    logger.info("ARGS: " + arg_str)


def label_command(args):
    load_data(args.dirs)

def load_data(dirs):
    print("INVOKED LABEL_DATA")

    # Load all of the states from pickles.
    data_path = "./v3_datasets/"

    datasets = []
    for d in dirs:
        path = data_path + d
        for idx, file in enumerate(os.listdir(path)):
            filename = os.path.join(path, file)
            with open(filename, 'rb') as fo:
                datasets.extend(pickle.load(fo, encoding='bytes'))

    # Transform into expected state.
    datasets = np.array(datasets)
    datasets = board_data(datasets)

    # Drop extraneous policy and q values.
    board_states = []
    for d in datasets:
        board_states.append(d[0])

    return np.array(board_states)

def label_data(board_states, balance=False):
    print("total boards: ", len(board_states))

    #attack_concept = get_attack_concept_label(board_states, can_make_four_consecutive_ones)
    #print("total attack: ", sum(attack_concept))
   
    attack_concept = opponent_has_double_attack(board_states)
    print("total attack: ", sum(attack_concept))

    #attack_concept = get_threat_concept_label(board_states, can_make_four_consecutive_ones)
    #print("total threat: ", sum(attack_concept))

    # Get all positive examples
    # Randomly sample an equal number of negative examples
    # Filter out the relevant board_states
    # Shuffle everything and return
    if balance:
        #positive = (attack_concept==1) | (threat_concept==1)
        #negative = (attack_concept==0) & (threat_concept==0)
        positive = attack_concept==1
        negative = attack_concept==0
        positive_indices = np.where(positive)[0]
        negative_indices = np.where(negative)[0]
        negative_indices = np.random.choice(negative_indices, len(positive_indices), replace=False)
        selection = np.concatenate((positive_indices, negative_indices))
        np.random.shuffle(selection)

        board_states = board_states[selection]
        attack_concept = attack_concept[selection]
        #threat_concept = threat_concept[selection]
        print("TOTAL BALANCED DATA: (board, pos, neg)", (len(board_states), len(positive_indices), len(negative_indices)))



    #player_double_attack = player_has_double_attack(board_states)
    #print("total threat: ", sum(player_double_attack))

    #opponent_double_attack = opponent_has_double_attack(board_states)
    #print("total threat: ", sum(opponent_double_attack))

    #return np.array([attack_concept, threat_concept, player_double_attack, opponent_double_attack]), board_states
    #return np.array([attack_concept, threat_concept]), board_states
    return np.array([attack_concept]), board_states


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("board_data", type=str, nargs='+', help="The directories containing the labeled board states.")
    run_parser.add_argument("--model_name", type=str, default="", help="The model file to load into the network.")
    run_parser.set_defaults(func=run)

    parser_label = subparsers.add_parser("label")
    parser_label.add_argument('dirs', type=str, nargs='+')
    parser_label.add_argument('--out', type=str)
    parser_label.set_defaults(func=label_command)

    parser_all = subparsers.add_parser("run_all")
    parser_all.set_defaults(func=run_all)

    args = parser.parse_args()


    args.func(args) 