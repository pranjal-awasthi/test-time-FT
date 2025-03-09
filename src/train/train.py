from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, PreTrainedModel, PreTrainedTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from src.data.data_utils import DataPoint, Solution


# adapted from: https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh
def finetune(dataset: Dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_length: int =1024, batch_size: int = 2, epochs: int = 3, learning_rate: float = 5e-4):

  # Split into training and validation sets
  train_size = int(0.9 * len(dataset))
  val_size = len(dataset) - train_size

  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


  train_dataloader = DataLoader(
              train_dataset,  
              sampler = RandomSampler(train_dataset), 
              batch_size = batch_size 
          )

  validation_dataloader = DataLoader(
              val_dataset, 
              sampler = SequentialSampler(val_dataset), 
              batch_size = batch_size 
          )


  # Tell pytorch to run this model on the GPU.
  device = torch.device("cuda")
  model.cuda()

  learning_rate = learning_rate
  warmup_steps = 100
  epsilon = 1e-8

  # this produces sample output every 100 steps
  sample_every = 100

  optimizer = AdamW(model.parameters(),
                    lr = learning_rate,
                    eps = epsilon
                  )
  # Total number of training steps is [number of batches] x [number of epochs].
  total_steps = len(train_dataloader) * epochs

  # Create the learning rate scheduler.
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps = warmup_steps,
                                              num_training_steps = total_steps)


  total_t0 = time.time()


  model = model.to(device)

  for epoch_i in range(0, epochs):

      # ========================================
      #               Training
      # ========================================

      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      print('Training...')

      t0 = time.time()

      total_train_loss = 0

      model.train()

      for step, batch in enumerate(train_dataloader):

          b_input_ids = batch[0].to(device)
          b_labels = batch[0].to(device)
          b_masks = batch[1].to(device)

          model.zero_grad()

          outputs = model(  b_input_ids,
                            labels=b_labels,
                            attention_mask = b_masks,
                            token_type_ids=None
                          )

          loss = outputs[0]

          batch_loss = loss.item()
          total_train_loss += batch_loss

          # Get sample every x batches.
          if step % sample_every == 0 and not step == 0:

              elapsed = format_time(time.time() - t0)
              print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

              model.eval()

              sample_outputs = model.generate(
                                      bos_token_id=random.randint(1,30000),
                                      do_sample=True,
                                      top_k=50,
                                      max_length = 200,
                                      top_p=0.95,
                                      num_return_sequences=1
                                  )
              for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

              model.train()

          loss.backward()

          optimizer.step()

          scheduler.step()

      # Calculate the average loss over all of the batches.
      avg_train_loss = total_train_loss / len(train_dataloader)

      # Measure how long this epoch took.
      training_time = format_time(time.time() - t0)

      print("")
      print("  Average training loss: {0:.2f}".format(avg_train_loss))
      print("  Training epoch took: {:}".format(training_time))

      # ========================================
      #               Validation
      # ========================================

      print("")
      print("Running Validation...")

      t0 = time.time()

      model.eval()

      total_eval_loss = 0
      nb_eval_steps = 0

      # Evaluate data for one epoch
      for batch in validation_dataloader:

          b_input_ids = batch[0].to(device)
          b_labels = batch[0].to(device)
          b_masks = batch[1].to(device)

          with torch.no_grad():

              outputs  = model(b_input_ids,
                              attention_mask = b_masks,
                              labels=b_labels)

              loss = outputs[0]

          batch_loss = loss.item()
          total_eval_loss += batch_loss

      avg_val_loss = total_eval_loss / len(validation_dataloader)

      validation_time = format_time(time.time() - t0)

      print("  Validation Loss: {0:.2f}".format(avg_val_loss))
      print("  Validation took: {:}".format(validation_time))

  print("")
  print("Training complete!")
  print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

  return model