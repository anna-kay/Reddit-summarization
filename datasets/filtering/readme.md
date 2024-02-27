The Reddit TIFU dataset that is accessible through the Hugging Face library by the name “reddit_tifu” comprise of a “long” and a “short” version. Just to note, “long” and “short” do not refer to the size of the dataset but the type of the data items. In this repo, the “long” version is used.

reddit_tifu = load_dataset('reddit_tifu', 'long', split='train')

Each of the data items of the dataset comprises of the following fields: 'ups', 'num_comments', 'upvote_ratio', 'score', 'documents', 'tldr', 'title'.
We focus on the fields 'documents' and 'tldr', as these are relevant to the task of summarization.


# Filtering steps applied in this notebook

Step 1: inspect Reddit TIFU for duplicates of the source texts ('documents' column).

Step 2: inspect dataset for problematic source texts ('documents' column).

Step 3: inspect dataset for problematic summaries ('tldr' column).

Step 4: Aggregate all the indices that should be removed, found so far.

Step 5: Remove the indices 
    •	 Reddit TIFU indices that correspond to duplicates
    •	 Reddit TIFU indices that will be removed from the dataset (duplicates + not useful)

Step 6:  inspect the rest of the dataset for duplicates of summaries (column 'tldr' -> 'clean_text'), i.e., find the candidate duplicates based on the 'tldr' column

Step 7:  Compare the corresponding source texts ('documents' column) to figure out if they are actual duplicates.
    •	to compare the source texts for similarity ROUGE-2 recall is used
    •	two texts are considered duplicates if ROUGE-2 recall > 0.8 
    •	this way of computing similarity is based on the approach used in *Zhang, J., Zhao, Y., Saleh, M., & Liu, P. (2020, November). Pegasus: Pre-training with        extracted gap-sentences for abstractive summarization. In International Conference on Machine Learning (pp. 11328-11339). PMLR.*

    
# Findings of the filtering process

Reddit TIFU does contain duplicates, more specifically:
1.	38 exact duplicates
2.	56 almost duplicates

It is worth noting that there is one data item that has both exact duplicates and almost duplicates within the dataset, comprising a total of 25 occurrences of its duplicates (25 duplicates & 1 original data point). Examples of such indices (order of the data point in the dataset) are 8200, 8207 and 8208.
 
