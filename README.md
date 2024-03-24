## Overview
This project revolves around the task of Abstractive Summarization, a Natural Language Processing (NLP) task.
Transformer-based models (deep learning) are used for the summarization of informal, noisy texts. The texts come from Reddit. 

All models checkpoints and datasets use in the project are downloaded from Hugging Face 🤗.

## Models
1. **BART**, https://arxiv.org/abs/1910.13461 (paper)
2. **PEAGSUS**, http://proceedings.mlr.press/v119/zhang20ae (paper), https://github.com/google-research/pegasus (github)
3. **ProphetNet**, https://arxiv.org/abs/2001.04063 (paper), https://github.com/microsoft/ProphetNet/tree/master/ProphetNet (github)

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

1. Overlap between Webis-TLDR-17 and Reddit TIFU <br><br>
As one can observe from the columns 'Subreddit' and 'Time Span', there is a potential for overlap between the two datasets, as both include data from the subreddit 'r/tifu' spanning from 2013 to 2016. This project investigates and confirms the presence of this overlap. 
The two datasets share approxiamtely 5,700 common items, which constitues 13.5% of Reddit TIFU, 10.9% of “r/tifu” items of Webis-TLDR-17, and 0.15% of the total of Webis-TLDR-17.


## Project Structure 
```
| - .vscode/
| - - launch.json
| - notebooks/
| - - EDA/
| - - filtering/
| - src/
| - - dataset.py
| - - train.py
| - - train_without_optimizer.py
| - - utils/
```
