# Transformer Assignment (PA2) - CSE 256

## Overview
This project implements various natural language processing models, focusing on the use of transformers for tasks such as classification and language modeling. The assignment is part of the CSE 256 course, and it involves training and evaluating a Transformer-based language model, a classifier, and a disentangled attention model on political speeches.

## Requirements
Ensure you have the following packages installed:

- `torch`
- `matplotlib`
- `argparse`

## Directory Structure
- **speechesdataset/**: Directory containing text data for training and testing. Ensure you have the following files in this directory:
  - `train_CLS.tsv` : Training data for the classifier
  - `test_CLS.tsv` : Testing data for the classifier
  - `train_LM.txt` : Training data for the language model
  - `test_LM_obama.txt`, `test_LM_wbush.txt`, `test_LM_hbush.txt` : Testing data for language modeling

- **Code Files**:
  - `tokenizer.py`: Implements `SimpleTokenizer` used for word tokenizing the text.
  - `dataset.py`: Contains `SpeechesClassificationDataset` and `LanguageModelingDataset` classes for data handling.
  - `transformer.py`: Defines transformer models including `Encoder`, `Decoder`, and `CLSModel`.
  - `disentangled_attention.py`: Implements `DisentangledDecoder`.
  - `utilities.py`: Contains helper functions. **Note:** This is modified a bit from the handed out version to support using hooks, as did the tasks cronologically, and didn't want to restructure my model after having a working implementation.

## Running the Code
The script allows you to run different parts of the NLP models. Make sure to navigate to the root folder before running `main.py`. Use the following command line arguments to control which part of the project to run. 

### Arguments
- **Model Selection**:
  - `--part1`: Run the classifier model.
  - `--part2`: Run the language model.
  - `--part3`: Run the disentangled language model.

- **Language Model Sub-configurations**:
  If no arguments are specified other than `--part2`, all models will train. If you only want to train specific models, use any if these flags:
  - `--train-lm`: For training on the train_LM.txt dataset.
  - `--obama`: For training on the test_LM_obama.txt dataset.
  - `--wbush`: For training on the test_LM_wbush.txt dataset.
  - `--hbush`: For training on the test_LM_hbush.txt dataset.

- **Plotting**:
  - `--show_plot`: Show plots.
  - `--save_plot`: Save plots to disk.

### Examples
1. To run the classifier model:
    ```sh
    python main.py --part1
    ```

2. To train the language model and run it on Obama speeches:
    ```sh
    python main.py --part2 --train-lm --obama
    ```

3. To train the disentangled language model and visualize the plots:
    ```sh
    python main.py --part3 --show_plot
    ```

## Key Hyperparameters
- **General Hyperparameters**:
  - `batch_size = 16`: Batch size for data loaders.
  - `block_size = 32`: Maximum context length for predictions.
  - `learning_rate = 1e-3`: Learning rate for the optimizer.
- **Transformer Hyperparameters**:
  - `n_embd = 64`: Embedding dimension.
  - `n_head = 2`: Number of attention heads.
  - `n_layer = 4`: Number of transformer layers.

## Training and Evaluation
- **Classifier Model**: Trains a single hidden layer feed-forward network using embeddings from the transformer encoder. Prints accuracy for both train and test sets after each epoch.
- **Language Model**: Trains a transformer-based decoder to model language, evaluates perplexity every 100 iterations.
- **Disentangled Model**: Uses a disentangled attention decoder to improve language modeling.

## Contact
Author: Mathias Otnes

Mail: Mathias.otnes@gmail.com

Feel free to reach out if you have any questions regarding the implementation or usage of the models.

