import os
import argparse
import torch
import random
import json
import numpy as np
import string

from src.data.data_utils import CustomDataset, ClusteringDataPoint, LineSchedulingDataPoint, MSTDataPoint
from src.models.model import load_model
from src.train.train import finetune
from src.algs import best_of_alg, best_of_llm, top_ift

def parse_args():
    parser = argparse.ArgumentParser(description='Iterated Finetuning')
    parser.add_argument('--config', type=str, default=None, help='Experiment config file location')
    parser.add_argument('--seed', type=int, default=0, help='Randoms seed for the experiment')
    
    args = parser.parse_args()
    
    return args

def sanity_check(args_dict):
    pass

def main():
    args = parse_args()

    seed = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # generate random experiment id
    characters = string.digits
    experiment_id  = ''.join(random.choice(characters) for _ in range(6))
    print(f"Experiment id: {experiment_id}")

    if not args.config:
        print("You must specify a config file.")
        return

    elif not os.path.exists(args.config):
        print(f"no config file found at location: {args.config}")
        return

    f = open(args.config)
    args_dict = json.load(f)
    sanity_check(args_dict)

    problem = args_dict["problem"]

    max_length = args_dict["tokenizer_params"][problem]["max_length"]

    model_dir = args_dict["model_dir"]

    # if model_dir is empty then instruction tune the base model
    if not model_dir:

        it_params = args_dict["it_params"][problem]
        problem_params = args_dict["problem_params"][problem]

        # generate finetuning data
        ft_data = []
        match problem:
            case "clustering":
                for _ in range(it_params["num_samples"]):
                    clustering_instance = ClusteringDataPoint(num_points = problem_params["num_points"], r = problem_params["r"])
                    clustering_instance.generate()
                    ft_data.append(clustering_instance.generate_random_ft_data_point())

            case "mst":
                for _ in range(it_params["num_samples"]):
                    mst_instance = MSTDataPoint(num_lines = problem_params["num_lines"], prob = problem_params["prob"], deg = problem_params["deg"])
                    mst_instance.generate()
                    ft_data.append(mst_instance.generate_random_ft_data_point())

            case "line_scheduling":
                for _ in range(it_params["num_samples"]):
                    line_instance = LineSchedulingDataPoint(num_points = problem_params["num_points"], travel_time_range = problem_params["travel_time_range"], box_constraint_range = problem_params["box_constraint_range"])
                    line_instance.generate()
                    ft_data.append(line_instance.generate_random_ft_data_point())
    

        model, tokenizer = load_model(args_dict["model_name"])
        ft_dataset = CustomDataset(tokenizer, ft_data, max_length)
        model = finetune(ft_dataset, model, tokenizer, max_length, it_params["batch_size"], it_params["epochs"], it_params["learning_rate"])
        device = torch.device("cuda")
        model.cuda()

        # save the model
        model_dir = os.path.join(args_dict["experiment_dir"], f"{experiment_id}")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        with open(os.path.join(model_dir, "model_weights.pth"), "wb") as f:
            torch.save(model.state_dict(), f)

        del model, tokenizer
    
    torch.cude.empty_cache()

    # generate test data
    test_data = []
    match problem:
        case "clustering":
            for _ in range(problem_params["num_test_points"]):
                clustering_instance = ClusteringDataPoint(num_points = problem_params["num_points"], r = problem_params["r"])
                clustering_instance.generate()
                test_data.append(clustering_instance)

        case "mst":
            for _ in range(it_params["num_test_points"]):
                mst_instance = MSTDataPoint(num_lines = problem_params["num_lines"], prob = problem_params["prob"], deg = problem_params["deg"])
                mst_instance.generate()
                test_data.append(mst_instance)

        case "line_scheduling":
            for _ in range(it_params["num_test_points"]):
                line_instance = LineSchedulingDataPoint(num_points = problem_params["num_points"], travel_time_range = problem_params["travel_time_range"], box_constraint_range = problem_params["box_constraint_range"])
                line_instance.generate()
                test_data.append(line_instance)


    algs_to_run = []
    if args_dict["algorithm"] == "all":
        algs_to_run = ["best_of_alg", "best_of_llm", "top_ift"]
    else:
        algs_to_run = [args_dict["algorithm"]]

    
    for alg in algs_to_run:
        # load model
        model, tokenizer = load_model(args_dict["model_name"])
        with open(os.path.join(model_dir, "model_weights.pth"), "rb") as f:
            model.load_state_dict(torch.load(f))
        model.cuda()

        eval_list = []

        match alg:
            case "best_of_alg":
                for instance in test_data:
                    evals = best_of_alg(instance, model, tokenizer, args_dict["alg_params"]["best_of_alg"][problem]["num_to_sample"], max_length)
                    eval_list.append(evals)

            case "best_of_llm":
                for instance in test_data:
                    evals = best_of_llm(instance, model, tokenizer, args_dict["alg_params"]["best_of_llm"][problem]["num_to_sample"])
                    eval_list.append(evals)

            case "top_ift":
                for instance in test_data:
                    evals = top_ift(instance, model, tokenizer, args_dict["alg_params"]["top_ift"][problem]["m"], args_dict["alg_params"]["top_ift"][problem]["M"], args_dict["alg_params"]["top_ift"][problem]["Q"], max_length, args_dict["alg_params"]["top_ift"][problem]["batch_size"], args_dict["alg_params"]["top_ift"][problem]["epochs"], args_dict["alg_params"]["top_ift"][problem]["learning_rate"])
                    eval_list.append(evals)

        # save the evals
        log_dir = os.path.join(args_dict["experiment_dir"], f"{experiment_id}/logs/{problem}/{alg}")
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        with open(os.path.join(log_dir, "evals.jsonl"), "w") as f:
            for evals in eval_list:
                f.write(json.dumps(evals))
                f.write("\n")


if __name__ == "__main__":
    main()