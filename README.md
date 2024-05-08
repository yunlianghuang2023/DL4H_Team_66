The code for the paper "BEHRT: Transformer for Electronic Health Records"

• Data Preprocessing:
	1. preprocess/data_parquet_make.ipynb: This notebook reads raw source files containing patient and admission data to generate patient diagnosis records in the data output file data.parquet. The output includes columns such as SUBJECT_ID, ICD9_CODE, age for each visit separated by a designated delimiter, age of each visit, and total visit count.
	2. preprocess/token2idx_make.ipynb: This notebook reads the raw source file D_ICD_DIAGNOSES.csv to create a mapping between diagnoses and ICD9_CODE, saving the result in the data output file vocab_token_idx.pkl.
• Data Loading:
	1. dataLoader/MLM.py: Defines the MLMLoader class responsible for loading and preparing data for training model input.
	2. dataLoader/utils.py: Contains helper functions utilized by dataLoader/MLM.py.
• Model Building:
	1. model/MLM.py: Defines three Bert model classes - BertEmbeddings, CustomBertModel, and BertForMaskedLM. The calling sequence is BertForMaskedLM -> CustomBertModel -> BertEmbeddings. These models are utilized for training the Bert model later.
	2. model/utils.py: Defines a helper function called age_vocab, which is utilized in MLM_UIUC.ipynb.
	3. model/optimiser.py: Defines an optimizer function called at MLM_UIUC.ipynb.
MLM_UIUC.ipynb: 
	This main file is responsible for training and evaluating Bert models. It encompasses data loading, reading, and validation, model training and evaluation, and presents the final results.![image](https://github.com/yunlianghuang2023/DL4H_Team_66/assets/139084839/1af37500-139b-4e11-848d-f1fd55224207)
