## Overview
This project revolves around the task of Abstractive Summarization, a Natural Language Processing (NLP) task.
Transformer-based models (deep learning) are used for the summarization of informal, noisy texts. The texts come from Reddit. 

The repo contains:
- Exploratory Data Analysis of the Reddit datasets
- Filtering of noise from the Reddit datasets
- Replication of the results of the papers that introduce the Transformer-based models for Abstractive Summarization
- Fine-tuning of the Transformer-based models on the Reddit datasets

All datasets and models checkpoints used in the project are downloaded from Hugging Face 🤗.

## Data

**Datasets:**
1. **Webis-TLDR-17**, https://aclanthology.org/W17-4508/ (paper), https://huggingface.co/datasets/webis/tldr-17 (🤗 dataset card)
2. **Reddit TIFU**, https://arxiv.org/abs/1811.00783 (paper), https://huggingface.co/datasets/reddit_tifu (🤗 dataset card)
<br>

| Dataset | Subreddit | Time Span | Size | Fields |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| Webis-TLDR-17 | 29,650 different subreddits ("r/tifu" included) | 2006-2016 | 3,848,330 | 'author', 'body', 'normalizedBody', 'subreddit', 'subreddit_id', 'id', **‘content’ (the source text), ‘summary’** |
| Reddit TIFU | "r/tifu" | Jan 2013 - Mar 2018 | 42,139 | 'ups', 'num_comments', 'upvote_ratio', 'score', 'documents', 'tldr', 'title', **‘documents’(the source text), ‘tldr’(the summary)** |
<br>

**Issues with the data:**

1. **Overlap between Webis-TLDR-17 and Reddit TIFU:** <br>
As one can observe from the columns 'Subreddit' and 'Time Span', there is a potential for overlap between the two datasets, as both include data from the subreddit 'r/tifu' spanning from 2013 to 2016. More specifically, Webis-TLDR-17 includes 52,219 items belonging to the "r/tifu" subreddit. This project investigates and confirms the presence of this overlap. The two datasets share approxiamtely 5,700 common items (exact matches after lowercasing & removing asterisks), which constitue 13.5% of Reddit TIFU, 10.9% of “r/tifu” items of Webis-TLDR-17, and 0.15% of the total of Webis-TLDR-17. - [overlap examination](https://github.com/anna-kay/Reddit-summarization/blob/main/notebooks/filtering/overlap_examination_Webis-TLDR-17_Reddit-TIFU_no_prior_filtering.ipynb)
2. **Both datasets contain duplicates:** <br>
**Webis-TLDR-17** contains 40,407 items that are exact duplicates of another item (30,966 non unique values), in terms of source text (‘content’ field), see <a href="https://github.com/anna-kay/Reddit-summarization/blob/main/notebooks/filtering/Webis-TLDR-17_filtering.ipynb">Webis-TLDR-17 filtering</a>.<br>
**Reddit TIFU** contains 38 items that are exact duplicates of another item (24 non unique values), in terms of source text ('documents' field) and 56 almost duplicates, see <a href="https://github.com/anna-kay/Reddit-summarization/blob/main/notebooks/filtering/Reddit-TIFU_filtering.ipynb">Reddit TIFU filtering</a>. It is worth noting that we detected an item that appears 25 times Reddit TIFU (25 exact or almost duplicates and one original, e.g. in positions 8200, 8207 and 8208 (indexes) of the dataset).
3. **No official train-val-test splits for either dataset:**<br>
No official train-val-test splits were found in the papers introducing or performing experiments on the datasets. Hugging Face Datasets also does not provide any splits. The entirety of both datasets was using the 'split='train' argument, like this:
```
webis_tldr = load_dataset('reddit', split='train')
reddit_tifu = load_dataset('reddit_tifu', 'long', split='train')
```
4. **Both datasets are noisy:**<br>
As Reddit is an open platform, data quality issues are expected. In the scope of the summarization task specifically, the most revelant issues are:
  - very short summaries not proportionate to the source text,
  - users not providing a summary in the summary field but instead posting a short message prompting to read the whole source text or the title, or providing a conlusion or a general truth, or posing a question.<br>
These issues render these data points actually not suitable for training summarization models due to their lack of coherence with summarization principles.


## Models
1. **BART**, https://arxiv.org/abs/1910.13461 (paper)
2. **PEGASUS**, http://proceedings.mlr.press/v119/zhang20ae (paper), https://github.com/google-research/pegasus (github)
3. **ProphetNet**, https://arxiv.org/abs/2001.04063 (paper), https://github.com/microsoft/ProphetNet/tree/master/ProphetNet (github)


## Dependencies

- **python** (v3.10)
- **torch** (v2.4.1)
- **transformers** (v4.44.1)
- **nltk** (v3.9.1)
- **sentencepiece** (v0.2.0)
- **scikit-learn** (v1.5.2)
- **matplotlib** (v3.9.2)
- **evaluate** (v0.4.1)
- **tqdm** (v4.66.5)
- **wandb** (v0.16.6)
- **bert-score** (v0.3.13)
- **sentence-transformers** (v3.0.1)
- **CUDA** (v11.8) – Required for GPU acceleration with PyTorch

## Project Structure 
```
| - .vscode/
| - - launch.json
| - notebooks/
| - - EDA/
| - - filtering/
| - - results_replication/
| - data/
| - - webis_tldr_mini/
| - - - webis_tldr_mini_train/
| - - - webis_tldr_mini_val/
| - - - webis_tldr_mini_test/
| - src/
| - - dataset.py
| - - train.py
| - - train_without_optimizer.py
| - - test.py
| - - utils/
```

* `.vscode` directory is only useful if the code is run in Visual Studio Code
*  `notebooks` contains the .ipynb files for the Exploratory Data Analysis, the filtering of the Reddit datasets, and the replication of the results of Abstractive Summarization for the BART, PEGASUS, and ProphetNet papers
*  `data` contains a small subset of the filtered WEBIS-TLDR-17, split into train, validation, and test (in .arrow fromat)
*  `src` contains the PyTorch code for the fine-tuning of the Transformer-based models on the Reddit datasets
    - `dataset.py`: defines the SummarizationDataset pytorch class for WEBIS-TLDR-17 data
    - `train.py`
    - `train_without_optimizer.py`
    - `test.py`
    - `utils`: contains the functions for getting the optimizer, training an epoch, evaluation an epoch, saving the best model, plotting train-validation losses, and comupting the evaluation metrics


 ## How to run

```
# Clone the repository
git clone https://github.com/anna-kay/Reddit-summarization.git

# Change into the project directory
cd Reddit-summarization/src

# Install dependencies

# Train
python train.py 

# Evaluate on testset
python test.py
```
