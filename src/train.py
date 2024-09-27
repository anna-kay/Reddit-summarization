# Standard library imports
import logging
import os
import random

# Third-party library imports
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader, Subset

from transformers import (
    ProphetNetTokenizer,
    ProphetNetForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    GenerationConfig
)

from transformers import get_linear_schedule_with_warmup

# Project-specific imports
from dataset import SummarizationDataset
from utils.utils import (get_parser,
                         get_optimizer,
                         plot_train_val_losses,
                         train_epoch, evaluate_epoch,
                         save_best_model,
                         calculate_metrics,
                         print_out_predictions_labels,
                         )


def main():

    os.environ['WANDB_DISABLED'] = 'true'

    webis_tldr_arguments = ["--data_dir", "data",
                            "--dataset_dir", "webis_tldr_mini",
                            "--train_dataset_dir", "webis_tldr_mini_train",
                            "--val_dataset_dir", "webis_tldr_mini_val",
                            "--checkpoint", "microsoft/prophetnet-large-uncased",
                            "--do_lower_case", "False",
                            "--max_source_length", "512",
                            "--max_target_length", "142",
                            "--batch_size", "2",
                            "--max_grad_norm", "1.0",
                            "--epochs", "3",
                            "--learning_rate", "1e-6",
                            "--epsilon", "1e-12",
                            # "--num_beams", "4",
                            # "--early_stopping", "True",
                            "--wandb_project", "Abstractive Summarization",
                            "--wandb_entity", "anna-kay"
                            ]

    parser = get_parser()

    args = parser.parse_args(webis_tldr_arguments)

    # Sequence (sentence) padding parameters
    max_source_length = args.max_source_length
    max_target_length = args.max_target_length

    # Training parameters
    batch_size = args.batch_size
    max_grad_norm = args.max_grad_norm
    epochs = args.epochs
    learning_rate = args.learning_rate
    epsilon = args.epsilon

    # Model parameters
    checkpoint = args.checkpoint
    do_lower_case = args.do_lower_case

    # WAndB
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity

    # Construct path to the dataset that will be used
    train_data_path = os.path.join(
        args.data_dir, args.dataset_dir, args.train_dataset_dir)
    val_data_path = os.path.join(
        args.data_dir, args.dataset_dir, args.val_dataset_dir)

    # Initialize WandB run
    wandb.init(project=wandb_project,
               entity=wandb_entity,
               # track hyperparameters and run metadata
               config={"learning_rate": learning_rate,
                       "architecture": checkpoint,
                       "dataset": "WEBIS-TLDR-17",
                       "epochs": epochs,
                       }
               )

    # Load train and validation data
    tokenizer = ProphetNetTokenizer.from_pretrained(checkpoint,
                                                    do_lower_case=do_lower_case)

    train_dataset = SummarizationDataset(train_data_path,
                                         tokenizer,
                                         max_source_length,
                                         max_target_length)

    # Sample train dataset
    train_subset = Subset(train_dataset, range(40))

    train_loader = DataLoader(train_subset,  # train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    val_dataset = SummarizationDataset(val_data_path,
                                       tokenizer,
                                       max_source_length,
                                       max_target_length)

    # Sample validation dataset
    val_subset = Subset(val_dataset, range(10))

    val_loader = DataLoader(val_subset,  # val_dataset
                            batch_size=batch_size)

    # Load the model
    model = ProphetNetForConditionalGeneration.from_pretrained(checkpoint)
    # Uncomment if not enough VRAM to enable gradient_checkpointing
    # model.gradient_checkpointing_enable()

    if torch.cuda.is_available():
        device = torch.device("cuda")

    model.to(device)

    # Set optimizer and scheduler
    optimizer = get_optimizer(model, learning_rate, epsilon)

    total_steps = len(train_loader)*epochs  # Total number of training steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Configure Generation Config, the values will affect only the generation
    generation_config = GenerationConfig.from_pretrained(checkpoint)

    generation_config.min_length=10
    generation_config.max_length=60
    generation_config.num_beams=4
    generation_config.no_repeat_ngram_size=3
    generation_config.length_penalty=2.0
    generation_config.early_stopping=True

    # Lists for the plot of training-validation loss
    train_loss_values, val_loss_values = [], []

    # Use in save_best_model to find & save the best model
    best_val_loss = float("inf")
    best_rouge_score = 0.0

    # List to store learning rates
    learning_rates = []

    # Training loop
    for epoch in range(epochs):

        epoch_count = epoch + 1

        print(
            f"\n---------------------------------------- Epoch {epoch_count}/{epochs} ----------------------------------------\n")

        # ------------------------ TRAINING PART ------------------------ #
        avg_train_loss, current_lr = train_epoch(model,
                                                 epoch,
                                                 train_loader,
                                                 optimizer,
                                                 max_grad_norm,
                                                 scheduler,
                                                 device,
                                                 wandb)

        learning_rates.append(current_lr)
        train_loss_values.append(avg_train_loss)

        print(f"Average train loss: {avg_train_loss: .3f}")

        # ------------------------ VALIDATION PART ------------------------ #
        avg_val_loss, predictions, labels = evaluate_epoch(model,
                                                                tokenizer,
                                                                epoch,
                                                                val_loader,
                                                                device,
                                                                generation_config,
                                                                wandb)
        val_loss_values.append(avg_val_loss)

        print(f"Average val loss: {avg_val_loss: .3f}")

        # Print out metrics for the epoch
        metrics = calculate_metrics(predictions, labels)
        rouge_L_sum = metrics['ROUGE']['rougeLsum']

        print(f"Metrics: {metrics}")
        wandb.log({"Metrics": metrics})

        print_out_predictions_labels(predictions, labels)

        # Check scores and store the best
        if avg_val_loss < best_val_loss:
            # Update the best validation loss
            best_val_loss = avg_val_loss
            # Store the best (according to val loss) model checkpoint & info
            save_best_model(model,
                            epoch_count,
                            folder="best_avg_val_loss",
                            best_metric="avg_val_loss")  # metric that is used to select best model

        if rouge_L_sum > best_rouge_score:
            # Update the best rouge score
            best_rouge_scores = metrics['ROUGE']
            # Store the best (according to rouge_L_sum) model checkpoint & info
            save_best_model(model,
                            epoch_count,
                            folder="best_rouge_score",
                            best_metric="rouge_score")  # metric that is used to select best model

    # Plots the tarining and validation losses of all the epochs
    plot_train_val_losses(train_loss_values, val_loss_values, epochs)

    # Log the final learning rates, log model artifacts & finish the WandB run
    wandb.log({"learning_rates_": learning_rates})
    wandb.log({"best_val_los": best_val_loss,
               "best_rouge_scores": best_rouge_scores})
    wandb.watch(model, log="all")
    wandb.finish()


if __name__ == "__main__":
    main()
