import torch
import os
import numpy as np
import random
import datetime
import evaluate
import gc
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from QTDataset import QTDataset
from openai_quantum import OpenAIGPTLMHeadModel, OpenAIGPTConfig


def train(lora_quantum, lora_rank, str_ent_layers):
    # HYPERPARAMETERS
    lora_alpha = 16

    generation_first_n_tokens = 5
    generation_length = 180
    generation_temperature = 0.75

    learning_rate = 8e-4
    batch_size = 12
    val_batch_size = 12
    val_every_n_epochs = 20
    max_epochs = 1000
    validation_prop = 0.1

    data_path = "./data/torchlight_2_parsed.pt"

    training_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    dir_name = (f"lora_quantum={lora_quantum}_"
                f"lora_rank={lora_rank}_"
                f"str_ent_layers={str_ent_layers}_timestamp={training_timestamp}")
    os.mkdir(f"./trained_models/{dir_name}")
    os.mkdir(f"./generated_examples/{dir_name}")

    data = torch.stack(list(torch.load(data_path).values()), dim=1)
    data_train, data_val = train_test_split(data, test_size=validation_prop)

    train_dataset = QTDataset(data_train)
    val_dataset = QTDataset(data_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    # Model loading and weight freezing (except lora)
    model_config = OpenAIGPTConfig(lora_quantum=lora_quantum,
                                   lora_rank=lora_rank,
                                   lora_alpha=lora_alpha,
                                   str_ent_layers=str_ent_layers)
    model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt", config=model_config)

    for p_name, p in model.named_parameters():
        if "lora" not in p_name:
            p.requires_grad = False
    model = model.cuda()

    # Optimizer, loss, writer etc.
    optimizer = Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter()

    # BERTScore
    bert_score_metric = evaluate.load("bertscore")
    tokenizer = AutoTokenizer.from_pretrained("openai-gpt", use_fast=False)

    decoded_val = tokenizer.batch_decode(val_dataset.inputs,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)

    # TRAINING LOOP

    best_bert_f1 = -1
    best_bert_filename = ""

    for epoch in range(0, max_epochs):
        train_running_loss = 0.0
        train_batches = 0

        model.train(True)
        for x, mask in tqdm(train_dataloader):
            optimizer.zero_grad()

            y = model(input_ids=x.cuda(), attention_mask=mask.cuda(), labels=x.cuda())
            loss = y.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 30.0)
            optimizer.step()

            train_running_loss += loss.item()
            train_batches += 1

        train_loss = train_running_loss / train_batches

        print(f"Epoch {epoch} - train loss: {train_loss}")

        writer.add_scalar("Loss/train", train_loss, epoch)

        del y

        if epoch % val_every_n_epochs == 0 or epoch == max_epochs - 1:
            val_running_loss = 0.0
            val_batches = 0
            val_examples = 0

            model.eval()
            os.mkdir(f"./generated_examples/{dir_name}/epoch={epoch}")
            with torch.no_grad():
                for x, mask in tqdm(val_dataloader):
                    y = model(input_ids=x.cuda(), attention_mask=mask.cuda(), labels=x.cuda())
                    loss = y.loss

                    val_running_loss += loss.item()

                    generated = model.generate(x.cuda()[:, :generation_first_n_tokens],
                                               max_length=generation_length,
                                               do_sample=True,
                                               temperature=generation_temperature)
                    generated_decoded = tokenizer.batch_decode(generated,
                                                               skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=True)

                    for generated_string in generated_decoded:
                        with open(f"./generated_examples/{dir_name}/epoch={epoch}/{val_examples}.txt", "w") as vf:
                            vf.write(generated_string)
                        val_examples += 1

                    s_done = val_batches * val_batch_size
                    bert_score_metric.add_batch(predictions=generated_decoded,
                                                references=decoded_val[s_done:s_done + x.shape[0]])

                    val_batches += 1

            torch.save(model.state_dict(), f'./trained_models/{dir_name}/{epoch}.pt')

            val_loss = val_running_loss / val_batches

            writer.add_scalar("Loss/validation", val_loss, epoch)

            bert_score = bert_score_metric.compute(lang='en')
            bert_score = {key: sum(value)/len(value) for key, value in bert_score.items() if type(value) is list}

            print(f"Epoch {epoch} - val loss: {val_loss} - BERTScore precision: {bert_score['precision']} - "
                  f"BERTScore recall: {bert_score['recall']} - BERTScore F1-score: {bert_score['f1']}")

            if bert_score['f1'] > best_bert_f1:
                best_bert_f1 = bert_score['f1']
                if best_bert_filename != "":
                    os.remove(best_bert_filename)
                best_bert_filename = f"./trained_models/{dir_name}/best_{epoch}.pt"
                torch.save(model.state_dict(), best_bert_filename)

            writer.add_scalar("BERTScore precision/validation", bert_score['precision'], epoch)
            writer.add_scalar("BERTScore recall/validation", bert_score['recall'], epoch)
            writer.add_scalar("BERTScore F1-score/validation", bert_score['f1'], epoch)

            del generated
            del y

        writer.flush()
        torch.cuda.empty_cache()
        gc.collect()

    writer.close()


if __name__ == '__main__':
    for i in range(3):
        for rank in [2, 4, 6]:
            for layers in [1, 2, 3]:
                train(True, rank, layers)
            train(False, rank, 1)
