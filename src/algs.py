from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, PreTrainedModel, PreTrainedTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from src.data.data_utils import DataPoint, Solution
import numpy as np

from src.data.data_utils import CustomDataset, DataPoint, Solution
from src.models.model import load_model, generate_from_model, get_log_prob
from src.train.train import finetune

def best_of_llm(instance: DataPoint, model: PreTrainedModel, tokenizer: PreTrainedModel, num_to_sample: int = 1) -> Dict[str, float]:

  query = instance.convert_to_prompt()

  best_cost = np.inf
  best_eval = {}

  while num_to_sample > 0:
    torch.cuda.empty_cache()
    samples = generate_from_model(instance, model, tokenizer, min(20, num_to_sample))
    num_to_sample -= min(20, num_to_sample)

    iters = 0
    for solution in samples:        
        evals = solution.evaluate()
        if(evals["total_cost"] < best_cost):
            print("***********************\n")
            print(f"Iteration: {iters}")
            print(f"total cost: {evals["total_cost"]}")
            best_cost = evals["total_cost"]
            best_eval = evals

    iters += 1
    del samples

  return best_eval


def best_of_alg(instance: DataPoint, model: PreTrainedModel, tokenizer: PreTrainedModel,  num_to_sample: int = 1, max_length: int = 768) -> Dict[str, float]:

    query = instance.convert_to_prompt()
    responses = []
    solutions = []

    for i in range(num_to_sample):
        response = instance.generate_random_feasible_solution()
        responses.append(response)
        solution.append(instance.parse_to_solution(query + "\n" + response))


    log_probs = get_log_prob(query, responses, model, tokenizer, max_length)

    indices = np.argsort(log_probs)

    return solution[indices[-1]].evaluate()

def top_ift(instance: DataPoint, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, m: int = 4, M: int = 10, Q: int = 12, max_length: int = 768, batch_size: int = 2, epochs: int = 3, learning_rate: float = 5e-4) -> Dict[str, float]:

    query = instance.convert_to_prompt()
    sample = generate_from_model(instance,  model, tokenizer, 1)
    if not sample:
        return {}
    
    best_eval = sample[0].evaluate()

    print(f"current best solution: {best_eval}")

    for i in range(Q):
        samples = generate_from_model(instance,  model, tokenizer, m*M)
        if not samples:
            return best_eval

        indices = np.argsort([s.evaluate["total_cost"] for s in samples])

        ft_dataset = []
        for j in range(m):
            ft_dataset.append({"query": query, "response": samples[indices[j]].get_response_string_from_solution()})

        custom_dataset = CustomDataset(tokenizer, ft_dataset, max_length)

        model = finetune(custom_dataset, model, tokenizer, max_length, batch_size, epochs, learning_rate)

        sample = generate_from_model(instance,  model, tokenizer, 1)

        if sample:
            evals = sample[0].evaluate()
            if evals["total_cost"] < best_eval["total_cost"]:
                best_eval = evals



    return best_eval
