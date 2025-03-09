import os
import argparse
import torch
import random
import json
import numpy as np
import string

import matplotlib.pyplot as plt

def plot_init():
    plt.figure(figsize=(5,5))
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.xlabel('Algorithm cost')
    plt.ylabel('LLM cost')

def plot_rect(x1, y1, x2, y2, color, label):
    plt.scatter(x1, y1, color=color, label=label)
    plt.fill_between([max(0,x1 - x2), x1 + x2], max(0,y1 - y2), y1 + y2, color=color, alpha=0.1)

def plot_show():
    plt.title(f'Pareto frontier')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Iterated Finetuning')
    parser.add_argument('--experiment_dir', type=str, default=None, help='Experiment dir')
    parser.add_argument('--experiment_id', type=str, default=None, help='Experiment id')
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()

    log_dir = os.path.join(args_dict["experiment_dir"], f"{experiment_id}/logs")

    for problem in ["clustering", "mst", "line_scheduling"]:
        if os.path.exists(os.path.join(log_dir, problem)):
            plt_vals = {}
            for alg in ["best_of_alg", "best_of_llm", "top_ift"]:
                if os.path.exists(os.path.join(log_dir, f"{problem}/{alg}/evals.jsonl")):
                    plt_vals[alg] = {}
                    with open(os.path.join(log_dir, f"{problem}/{alg}/evals.jsonl"), "r") as f:
                        for ell in f.readlines():
                            evals = json.loads(ell)
                            for k,v in ell.items():
                                if k in plt_vals[alg]:
                                    plt_vals[alg][k].append(v)
                                else:
                                    plt_vals[alg][k] = [v]


                    for k, v in plt_vals[alg].items():
                        plt_vals[alg][k+"_mean"] = np.mean(v)
                        plt_vals[alg][k+"_std"] = np.std(v)


            plot_init()
            color_list = ["blue", "black", "magenta", "orange", "green", "violet"]
            color_index = 0
            for alg in ["best_of_alg", "best_of_llm", "top_ift"]:
                if alg in plt_vals:
                    x_vals = []
                    e_vals = []
                    for k, v in plt_vals[alg].items():
                        if string.endswith(k, "_mean"):
                            x_vals.append(v)
                        elif string.endswith(k, "_std"):
                            e_vals.append(v)

                    plot_rect(x_vals[0], x_vals[1], e_vals[0], e_vals[1], color_list[color_index], alg)
                    color_index += 1

            plot_show()



if __name__ == "__main__":
    main()