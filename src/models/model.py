from typing import Dict, List, Tuple
import torch
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, PreTrainedModel, PreTrainedTokenizer
from src.data.data_utils import DataPoint, Solution

VALID_MODELS = ["gpt2"]

def load_model(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:

    if model_name not in VALID_MODELS:
        raise ValueError(f"model name must be one of the following: [{VALID_MODELS}]")


    match model_name:

        case "gpt2":

            tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

            configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

            model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def generate_from_model(instance: DataPoint, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, num_samples: int, max_tries: int = 10) -> List[Solution]:

  query = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<|startoftext|>'+ instance.convert_to_prompt() + "\n"))).unsqueeze(0).to(device)

  tries = 0
  while tries < max_tries:
    tries += 1

    sample_outputs = model.generate(
                            inputs=query,
                            do_sample=True,
                            top_k=50,
                            max_length = 1024,
                            top_p=0.95,
                            num_return_sequences=num_samples
                        )

    outputs = []
    try:
      for i in range(num_samples):
        output_str = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sample_outputs[i]))
        solution = instance.parse_to_solution(output_str)
        if solution.is_valid:
          outputs.append(solution)
        else:
          raise Exception("invalid solution")
      break
    except:
        continue

  return outputs if len(outputs) == num_samples else []

# get prob. of suffix from the LLM
def get_log_prob(query: str, responses: List[str], model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int) -> List[float]:

  pad_token_id = tokenizer.pad_token_id  #50258
  encodings_dict = tokenizer(query + "\n",
                              truncation=True, max_length=max_length, padding="do_not_pad")

  prefix_id_length = len(encodings_dict['input_ids'])

  log_prob_list = []

  num_to_sample = len(responses)
  curr_index = 0

  while num_to_sample > 0:
    # we do 2 at a time for a single Tesla V100 GPU setup. This can be increased if one has more GPUs.
    encodings_dict = tokenizer([query + "\n" + response for response in responses[curr_index:curr_index+2]],
                                truncation=True, max_length=max_length, padding="longest")

    curr_index += len(encodings_dict['input_ids'])
    num_to_sample -= len(encodings_dict['input_ids'])

    input_ids = torch.vstack(tuple([torch.tensor(t).to(device) for t in encodings_dict['input_ids']]))
    attention_mask = torch.vstack(tuple([torch.tensor(t).to(device) for t in encodings_dict['attention_mask']]))
    target = torch.vstack(tuple([torch.tensor(t[prefix_id_length:]).to(device) for t in encodings_dict['input_ids']]))

    target = target.unsqueeze(-1)

    outputs = model(  input_ids,
                      labels=input_ids,
                      attention_mask = attention_mask,
                      token_type_ids=None
                    )


    full_log_probs = torch.log(torch.nn.functional.softmax(outputs[1], dim=-1))[:,prefix_id_length-1:-1,:]

    mask = torch.tensor((torch.arange(0,full_log_probs.shape[-1]) != pad_token_id), dtype=torch.int32).to(device)

    mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(0)

    full_log_probs = full_log_probs * mask


    log_probs = torch.gather(full_log_probs, dim=-1, index=target).squeeze(-1)

    log_prob_list += list(torch.sum(log_probs, dim=-1).cpu().detach().numpy())

  return log_prob_list