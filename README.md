# GCN_impute

## File descriptions

### Small Dataset

1) ./data/sp500/sp500_price.csv: contains daily price data for 500 companies. (2518 × 500)

2) ./data/sp500/affMat.csv: correlation matrix based on dataset above (1), (487× 488). Last column contains labels.

3) ./data/sp500/fastMat.csv, googleMat.csv, gloveMat.csv: contains the known (p) embedding vectors out of 487 total companies. Last 2 columns contain the labels and word frequencies.


### Large Dataset

4) ./data/finance/aff.txt: 4092 companies and their daily price data. (4092 x 400)

5) ./data/finance/finance_token.csv: contains n_gram labels for companies.

6) ./data/finance/subfast.txt, subglove.txt, subgoogle.txt, contains around 7000 word embedding, both for company names and other embeddings.

7) ./data/finance/word_list.txt, not sure what this is, contains a list of 11347 words (?)

8) ./data/finance/gloveMat_4000.csv, googleMat_4000.csv, fastMat_4000.csv: (399 x 302) last two cols label and word freq. companies in glove.

9) ./data/finance/priceMat_4000.csv: (4092 x 401) price data for 4092 companies, labels in the last column.

10) ./data/finance/gloveMat_4000_uncommon.csv: o part

### Input for language model

11) ./data/GCN_embeds/lm_GCN_4000_<base_embed>_<delta_value>_<sigma_value>: embeddings learned by GCN, p+q+o

### Scripts

For running LSI, .LatentSemanticImputation/word_classification/mlr/imputation.py