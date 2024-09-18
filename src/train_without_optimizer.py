# Standard library imports
import logging
import os
import random

# Third-party library imports
from dataset import SummarizationDataset
from transformers import get_linear_schedule_with_warmup
from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import torch
import wandb

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')

# Project-specific imports
from utils.utils import get_parser, get_optimizer, train_epoch_manually_compute_grads, train_epoch, evaluate_epoch, compute_metrics, plot_train_val_losses

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,  # Set the logging level to DEBUG to log all messages
#     format='%(asctime)s - %(levelname)s - %(message)s',  # Define the format of log messages
#     filename='train.log'  # Specify the file where log messages will be written
# )


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
                            "--epochs", "2",
                            "--learning_rate", "1e-4",  # 5e-6
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
                       })

    # Load train and validation data
    tokenizer = ProphetNetTokenizer.from_pretrained(checkpoint,
                                                    do_lower_case=do_lower_case)

    train_dataset = SummarizationDataset(train_data_path,
                                         tokenizer,
                                         max_source_length,
                                         max_target_length)

    # Selects a subset of the train_dataset to speed up the experiments during debugging
    # Comment out when running the experiments
    # Indices of the samples you want to include in the subset
    subset_indices = list(range(80))
    train_dataset = Subset(train_dataset, subset_indices)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    val_dataset = SummarizationDataset(val_data_path,
                                       tokenizer,
                                       max_source_length,
                                       max_target_length)

    # Selects a subset of the train_dataset to speed up the experiments during debugging
    # Comment out when running the experiments
    # Indices of the samples you want to include in the subset
    subset_indices = list(range(20))
    val_dataset = Subset(val_dataset, subset_indices)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size)

    # Load the model
    model = ProphetNetForConditionalGeneration.from_pretrained(checkpoint)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    model.to(device)

    # Set optimizer and scheduler
    # optimizer = get_optimizer(model, learning_rate, epsilon)

    # optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)

    # total_steps = len(train_loader)*epochs # Total number of training steps
    # scheduler = get_linear_schedule_with_warmup(
    #     # optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=total_steps
    # )

    # Lists for the plot of training-validation loss
    train_loss_values, val_loss_values = [], []

    # Use in save_best_model to find & save the best model
    best_val_loss = float("inf")
    best_micro_avgs, best_macro_avgs = None, None

    # List to store learning rates
    learning_rates = []

    # Training loop
    for epoch in range(epochs):

        epoch_count = epoch + 1

        print(
            f"\n---------------------------------------- Epoch {epoch_count}/{epochs} ----------------------------------------\n")

        # ------------------------ TRAINING PART ------------------------#
        # train_epoch_manually_compute_grads does not take an optimizer as argument
        avg_train_loss = train_epoch_manually_compute_grads(model,
                                                            epoch,
                                                            train_loader,
                                                            max_grad_norm,
                                                            learning_rate,
                                                            device,
                                                            wandb)

        # learning_rates.append(current_lr)
        train_loss_values.append(avg_train_loss)

        print(f"Average train loss: {avg_train_loss: .3f}")
        print("\n")

        # ------------------------ VALIDATION PART ------------------------#
        avg_val_loss, predictions, labels = evaluate_epoch(model,
                                                           epoch,
                                                           val_loader,
                                                           device,
                                                           wandb)

        val_loss_values.append(avg_val_loss)
        print(f"Average val loss: {avg_val_loss: .3f}")
        print("\n")

        epoch_scores = compute_metrics(predictions, labels, tokenizer)
        print("\n")
        print(f"Epoch scores: {epoch_scores}")

        # TODO: add relevant (total) metrics
        # TODO: add wandb.logs

        # Check scores and store the best
        if avg_val_loss < best_val_loss:

            # Update the best metrics
            best_val_loss = avg_val_loss
            best_scores = epoch_scores

            # Store the best (according to val loss) model checkpoint & scores
            # save_best_model(model,
            #                 best_scores,
            #                 "early_stopping_model",
            #                 epoch_count)

    # Plots the tarining and validation losses of all the epochs
    plot_train_val_losses(train_loss_values, val_loss_values, epochs)

    # Save the last model checkpoint & scores
    # save_best_model(model,
    #                 best_scores,
    #                 "last_epoch_model",
    #                 epoch_count)

    # Log the final learning rates, log model artifacts & finish the WandB run
    # wandb.log({"learning_rates_": learning_rates})
    wandb.log({"best_scores": best_scores})
    wandb.watch(model, log="all")
    wandb.finish()


if __name__ == "__main__":
    main()
