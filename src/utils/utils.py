import argparse
import json
import os
from collections import Counter

import torch
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize
from evaluate import load

from tqdm import tqdm


def get_parser():
    
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--path", type=str, default = "data/APTNER_processed/", help="Directory of the data")
    parser.add_argument("--data_dir", type=str, default = "data", help="Directory of the data")
    parser.add_argument("--dataset_dir", type=str, default = "webis_tldr_mini", help="Directory of the dataset")
    parser.add_argument("--train_dataset_dir", type=str, default = "webis_tldr_mini_train", help="Directory for the train split of the dataset")
    parser.add_argument("--val_dataset_dir", type=str, default = "webis_tldr_mini_val", help="Directory for the validation split of the dataset")
    parser.add_argument("--checkpoint", type=str, default = "microsoft/prophetnet-large-uncased", help="Hugging Face model checkpoint")
    parser.add_argument("--do_lower_case", type=bool, default=False, help="True if the model is uncased, should be defined according to checkpoint")
    parser.add_argument("--max_source_length", type=int, default=512, help="Maximal number of tokens per sequence. All sequences will be cut or padded to this length.")
    parser.add_argument("--max_target_length", type=int, default=128, help="Maximal number of tokens per sequence. All sequences will be cut or padded to this length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=1e-12, help="Epsilon")
    parser.add_argument("--wandb_project", type=str, default="Abstractive Summarization", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="anna-kay", help="Wandb entity name")
    
    return parser

def get_optimizer(model, learning_rate, epsilon):
    
    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=epsilon)
    
    return optimizer


def train_epoch(model, epoch, train_loader, optimizer, scheduler, device, wandb): # max_grad_norm,
    
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader, desc=f"{epoch+1}"):
        input_ids = batch["input_ids"].to(device).long()
        attention_mask = batch["attention_mask"].to(device).long()
        labels = batch["labels"].to(device).long()

        # Zero gradients
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids,
                        # token_type_ids=None,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss
        loss.backward()
        train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

    # Calculate and log average training loss and learning rate for the epoch
    avg_train_loss = train_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]
    
    wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})
    wandb.log({"epoch": epoch+1, "learning_rate": current_lr})
    
    return avg_train_loss, current_lr


def train_epoch_manually_compute_grads(model, epoch, train_loader, learning_rate, device, wandb): # max_grad_norm,
    
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader, desc=f"{epoch+1}"):
        input_ids = batch["input_ids"].to(device).long()
        attention_mask = batch["attention_mask"].to(device).long()
        labels = batch["labels"].to(device).long()

        # Zero gradients
        model.zero_grad()

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss
        loss.backward()
        train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters()) #, max_norm=max_grad_norm)
        
        # Manually update model parameters
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad 

    # Calculate and log average training loss and learning rate for the epoch
    avg_train_loss = train_loss / len(train_loader)
    
    wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})
    # wandb.log({"epoch": epoch+1, "learning_rate": current_lr})
    
    return avg_train_loss


def evaluate_epoch(model, epoch, val_loader, device, wandb): 
    model.eval()
    val_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device).long()
            attention_mask = batch["attention_mask"].to(device).long()
            labels = batch["labels"].to(device).long()

            outputs = model(input_ids=input_ids,
                            # token_type_ids=None,
                            attention_mask=attention_mask,
                            labels=labels)

            logits = outputs.logits.to('cpu').numpy()   # logits = outputs.logits.detach().cpu().numpy()
                                                    # .detach() is redundant
            # TODO: Ensure that your model’s logits are in the shape (batch_size, sequence_length, vocab_size).
            label_ids = labels.to('cpu').numpy()

            val_loss += outputs.loss.item() # outputs.loss.mean().item()

            # Compute predicted labels from logits
            batch_predictions = np.argmax(logits, axis=2)
            predictions.extend(batch_predictions.tolist())
            true_labels.extend(label_ids.tolist())
            
            # TODO: clarify that this is *batch* val loss?
            # wandb.log({"epoch": epoch+1, "batch_val_loss": val_loss})
    
    avg_val_loss = val_loss/len(val_loader)
    wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss})
    
    return avg_val_loss, predictions, true_labels


def compute_metrics(predictions, labels, tokenizer):

    rouge_score = load("rouge")
    
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


def plot_train_val_losses(train_loss_values, val_loss_values, epochs):
    
    x = range(1, epochs+1)
    
    plt.title("Training & Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.xticks(x)
    plt.plot(x, train_loss_values, marker='o', label='train loss')
    plt.plot(x, val_loss_values, marker='o', label='valid loss')
    plt.legend()
    plt.grid(linestyle = '--')
    plt.show()