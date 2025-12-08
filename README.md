This directory contains the models implemented as extensions of DeepCoNN, from the 2017 paper "Joint deep modeling of users and items using reviews for recommendation" by Lei Zheng, Vahid Noroozi, and Philip S Yu.

Each model is self-contained and can be run separately.

# DeepCoNN-BERT
To run, install the packages listed in requirements.txt. Conda may be needed to install pytorch instead of pip. 
Preprocessed data and training/test splits are available in the data folder but may be re-processed and split by navigating to the data folder and running
```
python preprocess.py
```

Then navigate back to the main directory and run
```
python main.py
```
# DeepCoNN-LSTM
To run, navigate to the LSTM Recomender Notebook and open it in either google colab or jupyter notebook. Running all cells will train and test both DeepCoNN Baseline and the LSTM. All runs for this model were executed on a A100 CoLab GPU.
Preprocessed data and training/test splits are available in the data foler inside of the LSTM folder.

# SBERT
To run, open the juypter notebook and run the file from the top. Running all cells will train and test the SBERT. All runs were done on a M3 Pro. 

# DeepCoNN-RoBERTa
To run, install necessary packages first.
Run preprocess.py in the data folder to regenerate data splits as needed. Default data splits are available as csv files in data/music.
Then navigate back to main directory and run main.py
Changes to parameters can be made in config.py
