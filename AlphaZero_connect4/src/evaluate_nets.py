# -*- coding: utf-8 -*-
"""

"""
from MCTS_c4 import run_MCTS
from train_c4 import train_connectnet
from evaluator_c4 import evaluate_nets
from argparse import ArgumentParser
from alpha_net_c4 import ConnectNet, SmallConnectNet
import logging
import os
import torch
import os.path
import torch
import numpy as np
from connect_board import board as cboard
import encoder_decoder_c4 as ed
import copy
from MCTS_c4 import UCT_search, do_decode_n_move_pieces, get_policy
import pickle
import torch.multiprocessing as mp
import datetime
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def save_as_pickle(filename, data):
    completeName = os.path.join("./evaluator_data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
def load_pickle(filename):
    data = None
    completeName = os.path.join("./evaluator_data/",\
                                filename)
    try:
        with open(completeName, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    except Exception as e:
        logger.error(f"An error occurred in load_pickle for fileName: {completeName}, error: {e}")

    return data

class arena():
    def __init__(self, current_cnet, best_cnet, num_rollouts):
        self.current = current_cnet
        self.best = best_cnet
        self.cuda = torch.cuda.is_available()
        self.rollouts = num_rollouts
    
    def play_round(self):
        logger.info("Starting game round...")
        if np.random.uniform(0,1) <= 0.5:
            white = self.current; black = self.best; w = "current"; b = "best"
        else:
            white = self.best; black = self.current; w = "best"; b = "current"
        current_board = cboard()
        checkmate = False
        dataset = []
        value = 0; t = 0.1
        while checkmate == False and current_board.actions() != []:
            dataset.append(copy.deepcopy(ed.encode_board(current_board)))
            print(""); print(current_board.current_board)
            if current_board.player == 0:
                root = UCT_search(current_board,self.rollouts,white,t, self.cuda)
                policy = get_policy(root, t); print("Policy: ", policy, "white = %s" %(str(w)))
            elif current_board.player == 1:
                root = UCT_search(current_board,self.rollouts,black,t, self.cuda)
                policy = get_policy(root, t); print("Policy: ", policy, "black = %s" %(str(b)))
            current_board = do_decode_n_move_pieces(current_board,\
                                                    np.random.choice(np.array([0,1,2,3,4,5,6]), \
                                                                     p = policy)) # decode move and move piece(s)
            if current_board.check_winner() == True: # someone wins
                if current_board.player == 0: # black wins
                    value = -1
                elif current_board.player == 1: # white wins
                    value = 1
                checkmate = True
        dataset.append(ed.encode_board(current_board))
        if value == -1:
            dataset.append(f"{b} as black wins")
            return b, dataset
        elif value == 1:
            dataset.append(f"{w} as white wins")
            return w, dataset
        else:
            dataset.append("Nobody wins")
            return None, dataset
    
    def evaluate(self, num_games, cpu):
        current_wins = 0
        logger.info("[CPU %d]: Starting games..." % cpu)
        for i in range(num_games):
            with torch.no_grad():
                winner, dataset = self.play_round(); print("%s wins!" % winner)
            if winner == "current":
                current_wins += 1
            save_as_pickle("evaluate_net_dataset_cpu%i_%i_%s_%s" % (cpu,i,datetime.datetime.today().strftime("%Y-%m-%d"),\
                                                                     str(winner)),dataset)
        print("Current_net wins ratio: %.5f" % (current_wins/num_games))
        save_as_pickle("wins_eval_cpu_%i" % (cpu),\
                                             {"best_win_ratio": current_wins/num_games, "num_games":num_games})
        logger.info("[CPU %d]: Finished arena games!" % cpu)
        
def fork_process(arena_obj, num_games, cpu): # make arena picklable
    arena_obj.evaluate(num_games, cpu)


def evaluate_nets(args) :
    logger.info("Loading nets...")
    current_net_filename = os.path.join("./model_data/",\
                                    args.net_1)
    best_net_filename = os.path.join("./model_data/",\
                                    args.net_2)
    
    
    current_cnet = ConnectNet()
    if "small" in current_net_filename:
        current_cnet = SmallConnectNet()
    best_cnet = ConnectNet()
    if "small" in best_net_filename:
        best_cnet = SmallConnectNet()
    cuda = torch.cuda.is_available()
    if cuda:
        current_cnet.cuda()
        best_cnet.cuda()
    
    if not os.path.isdir("./evaluator_data/"):
        os.mkdir("evaluator_data")
    
    if args.MCTS_num_processes > 1:
        mp.set_start_method("spawn",force=True)
        
        current_cnet.share_memory(); best_cnet.share_memory()
        current_cnet.eval(); best_cnet.eval()
        
        checkpoint = torch.load(current_net_filename)
        current_cnet.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(best_net_filename)
        best_cnet.load_state_dict(checkpoint['state_dict'])
         
        processes = []
        if args.MCTS_num_processes > mp.cpu_count():
            num_processes = mp.cpu_count()
            logger.info("Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
        else:
            num_processes = args.MCTS_num_processes
        logger.info("Spawning %d processes..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=fork_process,args=(arena(current_cnet,best_cnet, args.num_rollouts), args.num_evaluator_games, i))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
               
        wins_ratio = 0.0
        for i in range(num_processes):
            stats = load_pickle("wins_eval_cpu_%i" % (i))
            wins_ratio += stats['best_win_ratio']
        wins_ratio = wins_ratio/num_processes
        print("Win Ratio {}".format(wins_ratio))
            
    elif args.MCTS_num_processes == 1:
        current_cnet.eval(); best_cnet.eval()
        checkpoint = torch.load(current_net_filename)
        current_cnet.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(best_net_filename)
        best_cnet.load_state_dict(checkpoint['state_dict'])
        arena1 = arena(current_cnet=current_cnet, best_cnet=best_cnet)
        arena1.evaluate(num_games=args.num_evaluator_games, cpu=0)
        
        stats = load_pickle("wins_eval_cpu_%i" % (0))
        print("Win Ratio {}".format(stats['best_win_ratio']))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--net_1", default="c4_current_net_trained_iter8.pth.tar")
    parser.add_argument("--net_2", default="cc4_small_current_net__iter5.pth.tar")
    parser.add_argument("--MCTS_num_processes", type=int, default=5, help="Number of processes to run MCTS self-plays")
    parser.add_argument("--num_games_per_MCTS_process", type=int, default=120, help="Number of games to simulate per MCTS self-play process")
    parser.add_argument("--num_evaluator_games", type=int, default=100, help="No of games to play to evaluate neural nets")
    parser.add_argument("--num_rollouts", type=int, default=777, help="The number of MCTS rollouts to perform during simulation step.")

    args = parser.parse_args()

    evaluate_nets(args)




