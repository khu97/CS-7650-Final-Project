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
