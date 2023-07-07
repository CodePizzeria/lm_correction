# Language Models for improving EMG decoded typing

This repository contains a notebook that will train a language model (LM), and another that will combine the LM softmax and EMG decoder softmax, perform a BEAM search, in order to output a corrected output sequence. 

## Getting Started

To set up the environment in conda, run the following command:
```
conda install pytorch=*=*cuda* cudatoolkit=11.1 seaborn ipykernel python-levenshtein tabulate -c pytorch
```

Evaluate the cells in model_training.ipynb to train a LM. 

Evaluate the cells in model_evaluation.ipynb to compare the performance of simulated EMG performance alone vs with LM integration and BEAM search. 

Copy model_evaluation.ipynb to integrate with real EMG decoding softmaxes. 
