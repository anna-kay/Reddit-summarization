import argparse
import json
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

import torch
import torch.optim as optim

def get_parser():
    
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--path", type=str, default = "data/APTNER_processed/", help="Directory of the data")
    parser.add_argument("--data_dir", type=str, default = "data", help="Directory of the data")
    parser.add_argument("--dataset_dir", type=str, default = "webis_tldr_mini", help="Directory of the dataset")
    parser.add_argument("--dataset_filepath", type=str, default = "webis_tldr_mini_train", help="Filepath for the train split of the dataset")
    parser.add_argument("--checkpoint", type=str, default = "microsoft/prophetnet-large-uncased", help="Hugging Face model checkpoint")
    parser.add_argument("--do_lower_case", type=bool, default=False, help="True if the model is uncased, should be defined according to checkpoint")
    parser.add_argument("--max_source_length", type=int, default=512, help="Maximal number of tokens per sequence. All sequences will be cut or padded to this length.")
    parser.add_argument("--max_target_length", type=int, default=128, help="Maximal number of tokens per sequence. All sequences will be cut or padded to this length.")
    parser.add_argument("--pad_token", type=str, default="PAD", help="Token to pad sequences to maximal length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=1e-12, help="Epsilon")
    parser.add_argument("--wandb_project", type=str, default="Abstractive Summarization", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="anna-kay", help="Wandb entity name")
    
    return parser


def train_epoch(model, epoch, train_loader, optimizer, max_grad_norm, scheduler, device, wandb):
    
    model.train()
    train_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device).long()
        attention_mask = batch["attention_mask"].to(device).long()
        labels = batch["labels"].to(device).long()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        token_type_ids=None,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss
        loss.backward()
        train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        
    # Calculate ang log average training loss and learning rate for the epoch
    avg_train_loss = train_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]
    
    wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})
    wandb.log({"epoch": epoch+1, "learning_rate": current_lr})
    
    return avg_train_loss, current_lr


def evaluate_epoch(model, epoch, val_loader, device, wandb):
   
    model.eval()
    val_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            input_ids = input_ids.to(device).long()
            attention_mask = attention_mask.to(device).long()
            labels = labels.to(device).long()

            outputs = model(input_ids=input_ids,
                            token_type_ids=None,
                            attention_mask=attention_mask,
                            labels=labels)

            logits = outputs.logits.to('cpu').numpy()   # logits = outputs.logits.detach().cpu().numpy()
                                                    # .detach() is redundant
            label_ids = labels.to('cpu').numpy()

            val_loss += outputs.loss.item() # outputs.loss.mean().item()

            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

            true_labels = [[int(val) for val in sublist] for sublist in true_labels]
            
            # TODO: clarify that this is *batch* val loss?
            # wandb.log({"epoch": epoch+1, "batch_val_loss": val_loss})
    
    avg_val_loss = val_loss/len(val_loader)
    wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss})
    
    return avg_val_loss, predictions, true_labels


def print_epoch_scores(valid_tags, pred_tags, labels):

    # TODO: change code, add ROUGE scores
    
    F1_SCORE = f1_score(valid_tags, pred_tags, average = None, labels=labels)
    PRECISION = precision_score(valid_tags,pred_tags, average = None, labels=labels)
    RECALL = recall_score(valid_tags,pred_tags, average = None, labels=labels)

    print("\nLabel\t\t F1-Score Precision Recall")
    for i in range(len(labels)):
        print("{0:15}\t {1}\t {2}\t {3}".format(labels[i], \
                                                round(F1_SCORE[i],2), \
                                                round(PRECISION[i],2), \
                                                round(RECALL[i],2)))

    return 0  

