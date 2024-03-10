# Standard library imports
import logging
import os
import random

# Third-party library imports
import wandb
import torch
from torch.utils.data import DataLoader, Subset

from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration
from transformers import get_linear_schedule_with_warmup

# Project-specific imports
from dataset import SummarizationDataset
from utils.utils import get_parser, get_optimizer, train_epoch, evaluate_epoch, print_epoch_scores
                    
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
                            "--learning_rate", "1e-4",
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
    train_data_path = os.path.join(args.data_dir, args.dataset_dir , args.train_dataset_dir)
    val_data_path = os.path.join(args.data_dir, args.dataset_dir , args.val_dataset_dir)
    
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
    
    print(train_dataset)

    print(train_dataset[5])

    # Assuming 'dataset' is your original dataset
    subset_indices = list(range(2800))  # Indices of the samples you want to include in the subset
    my_subset = Subset(train_dataset, subset_indices)

    print(my_subset[5])


    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)
    
    val_dataset = SummarizationDataset(val_data_path, 
                                       tokenizer, 
                                       max_source_length,
                                       max_target_length)
    
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size)
       
    # Load the model
    model = ProphetNetForConditionalGeneration.from_pretrained(checkpoint)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
    model.to(device)
    
    # Set optimizer and scheduler
    optimizer = get_optimizer(model, learning_rate, epsilon)
    
    total_steps = len(train_loader)*epochs # Total number of training steps
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
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
        
        print(f"\n---------------------------------------- Epoch {epoch_count}/{epochs} ----------------------------------------\n")    
        
        # ------------------------ TRAINING PART ------------------------#
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
    
        # ------------------------ VALIDATION PART ------------------------#
        avg_val_loss, predictions, labels = evaluate_epoch(model, 
                                                                epoch, 
                                                                val_loader, 
                                                                device, 
                                                                wandb)
        val_loss_values.append(avg_val_loss)
        
        print(f"Average val loss: {avg_val_loss: .3f}")
            

        
        # Print out scores for the epoch
        print_epoch_scores(labels, predictions)          
        
        # TODO: add relevant (total) metrics
        # TODO: add wandb.logs
        
        # Check scores and store the best
        if avg_val_loss < best_val_loss:
            
            # Update the best metrics
            best_val_loss = avg_val_loss
            # TODO: modify 
            best_micro_avgs = micro_avgs 
            best_macro_avgs = macro_avgs
            
            # Store the best (according to val loss) model checkpoint & scores
            save_best_model(model,
                            # TODO: modify
                            micro_avgs,
                            macro_avgs,
                            "early_stopping_model", 
                            epoch_count)
    
    # Plots the tarining and validation losses of all the epochs
    plot_train_val_losses(train_loss_values, val_loss_values, epochs)          
    
    # Save the last model checkpoint & scores
    save_best_model(model,
                     # TODO: modify
                    micro_avgs, 
                    macro_avgs,
                    "last_epoch_model",
                    epoch_count)
    
    # Log the final learning rates, log model artifacts & finish the WandB run
    # wandb.log({"learning_rates_": learning_rates})
     # TODO: modify
    wandb.log({"best_micro_avgs": best_micro_avgs, \
               "best_macro_avgs": best_macro_avgs})
    wandb.watch(model, log="all")
    wandb.finish()
    
    
if __name__ == "__main__":
    main() 