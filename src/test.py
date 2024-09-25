# Standard library imports
import logging
import os
import random
import statistics

# Third-party library imports
import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from transformers import (
    ProphetNetTokenizer,
    ProphetNetForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    GenerationConfig
)

from sentence_transformers import SentenceTransformer, util
from evaluate import load

# Project-specific imports
from dataset import SummarizationDataset
from utils.utils import get_parser, compute_rouge_metrics


def main():

    os.environ['WANDB_DISABLED'] = 'true'

    webis_tldr_arguments = ["--data_dir", "data",
                            "--dataset_dir", "webis_tldr_mini",
                            "--test_dataset_dir", "webis_tldr_mini_test",
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
    test_data_path = os.path.join(
        args.data_dir, args.dataset_dir, args.test_dataset_dir)

    # Initialize WandB run
    wandb.init(project=wandb_project,
               entity=wandb_entity,
               # track hyperparameters and run metadata
               config={"learning_rate": learning_rate,
                       "architecture": checkpoint,
                       "dataset": "WEBIS-TLDR-17",
                       # "epochs": epochs,
                       })

    # Load train and validation data
    tokenizer = ProphetNetTokenizer.from_pretrained(checkpoint,
                                                    do_lower_case=do_lower_case)

    test_dataset = SummarizationDataset(test_data_path,
                                        tokenizer,
                                        max_source_length,
                                        max_target_length)

    # Sample test dataset
    test_subset = Subset(test_dataset, range(50))

    test_loader = DataLoader(test_subset,  # test_dataset
                             batch_size=batch_size)

    print("\n---------------------------------------- Testing ----------------------------------------\n")

    # Load best model
    print("Loading model...")

    # TODO: replace the path of the model
    base_directory = os.path.abspath("best_avg_val_loss")
    model_directory = "best_model"
    path = os.path.join(base_directory, model_directory)

    checkpoint = "microsoft/prophetnet-large-uncased"

    model = ProphetNetForConditionalGeneration.from_pretrained(checkpoint)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)

    model.eval()

    # Configure Generation Config, the values will affect only the generation
    generation_config = GenerationConfig.from_pretrained(checkpoint)

    generation_config.min_length=10
    generation_config.max_length=60
    generation_config.num_beams=4
    generation_config.no_repeat_ngram_size=3
    generation_config.length_penalty=2.0
    generation_config.early_stopping=True

    test_loss = 0.0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device).long()
            attention_mask = batch["attention_mask"].to(device).long()
            labels = batch["labels"].to(device).long()

            outputs = model(input_ids=input_ids,
                            # token_type_ids=None,
                            attention_mask=attention_mask,
                            labels=labels)

            # logits = outputs.logits.detach().cpu().numpy()
            logits = outputs.logits.to('cpu').numpy()
            # .detach() is redundant
            # TODO: Ensure that your model’s logits are in the shape (batch_size, sequence_length, vocab_size).
            label_ids = labels.to('cpu').numpy()

            test_loss += outputs.loss.item()  # outputs.loss.mean().item()

            # Use model.generate() with beam search decoding to generate predictions for summaries
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )

            # Decode the generated sequences to text
            decoded_preds = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)

            # Replace -100 in the labels as they cannot be decoded
            label_ids = np.where(
                label_ids != -100, label_ids, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(
                label_ids, skip_special_tokens=True)

            # Append to the list for ROUGE score calculation
            predictions.extend(decoded_preds)
            true_labels.extend(decoded_labels)

    # Print average test loss
    avg_test_loss = test_loss/len(test_loader)
    print(f"\n\nAverage test loss: {avg_test_loss: .3f}")

    # Print out ROUGE scores for the test set
    rouge_metrics = compute_rouge_metrics(predictions, true_labels)
    print(f"\n\nROUGE scores: {rouge_metrics}")

    # Computation of Semantic Similarity using SBERT
    # Step 1: Load the pre-trained model
    semantic_similarity_model =  SentenceTransformer('all-mpnet-base-v2')
    # semantic_similarity_model =  SentenceTransformer('all-MiniLM-L6-v2')

    semantic_similarities = []
    semantic_similarity_min = float("inf")

    # Step 2: Define the terms
    for sent1, sent2 in zip(predictions, true_labels):

        # Step 3: Encode the terms into embeddings
        embedding1 = semantic_similarity_model.encode(sent1, convert_to_tensor=True)
        embedding2 = semantic_similarity_model.encode(sent2, convert_to_tensor=True)

        # Step 4: Compute the cosine similarity
        similarity = float(util.cos_sim(embedding1, embedding2))

        if similarity < semantic_similarity_min:
            semantic_similarity_min = similarity

        semantic_similarities.append(similarity)        

    semantic_similarity_avg = sum(semantic_similarities)/len(semantic_similarities)

    print(f"\n\nSBERT Semantic similarity average: {semantic_similarity_avg}")
    print(f"SBERT Semantic similarity minimum: {semantic_similarity_min}")

    # BERTScore 
    bertscore = load("bertscore")
    bertscore_metrics = bertscore.compute(predictions=predictions, references=true_labels, lang="en")

    bertscore_metrics_avgs = {"precision": statistics.mean(bertscore_metrics["precision"]),
                              "recall": statistics.mean(bertscore_metrics["recall"]),
                              "f1": statistics.mean(bertscore_metrics["f1"])}

    print(f"\n\nBERTScore metrics: {bertscore_metrics_avgs}")

if __name__ == "__main__":
    main()
