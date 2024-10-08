# Self-Correction Language Model Training

This project is a Pytorch and Transformers implementation of the SCoRe (Self-Correction via Reinforcement Learning) method for training language models to improve their self-correction abilities over multiple attempts. The training regime is based on the paper **["Training Language Models to Self-Correct via Reinforcement Learning"](https://arxiv.org/abs/2409.12917)**.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Output Files](#output-files)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Overview

The script trains a language model to improve its responses over multiple turns by self-correcting based on its own generated outputs. It follows a two-stage reinforcement learning process:

1. **Stage I**: Fine-tunes the model to improve the second attempt while keeping the first attempt close to the base model's outputs.
2. **Stage II**: Trains the model using multi-turn RL to optimize both attempts, with a reward bonus to promote effective self-correction.

## Installation

Ensure you have Python 3.10 or higher installed. Install the required libraries:

```bash
pip install torch transformers datasets accelerate tqdm
```

## Usage

Run the script using Python:

```bash
python main.py [--arguments]
```

## Arguments

- `--model_name`: Name of the model to train (default: `'Qwen/Qwen2.5-1.5B-Instruct'`).
- `--dataset_name`: Name of the dataset to use (default: `'lighteval/MATH'`).
- `--question_column`: Column name for the questions in the dataset (default: `'problem'`).
- `--answer_column`: Column name for the gold standard answers in the dataset (default: `'solution'`).

Example:

```bash
python main.py --model_name 'Qwen/Qwen2.5-1.5B-Instruct' --dataset_name 'lighteval/MATH' --question_column 'problem' --answer_column 'solution'
```

## Output Files

- `accuracies.txt`: Contains the accuracies of the first and second attempts during training and evaluation.
- `generations.txt`: Contains the prompts and generations (first and second attempts) during training and evaluation.

## Experiment Results

We evaluated the implemented training pipeline on Qwen/Qwen2.5-1.5B-Instruct. During the training, the model was evaluated at different global steps. The following plot visualizes the model's first and second attempt accuracies over the course of training.

![Model Test Accuracy at Different Global Steps](result.png)

The dotted horizontal lines represent the base model's original accuracies:

- Base First Attempt Accuracy: **71%**
- Base Second Attempt Accuracy: **75%**

As shown in the graph, the second attempt accuracy significantly improves at most stages during the training process, reaching as high as **88%** at certain points. However, the results exhibit some instability, with both first and second attempt accuracies fluctuating over different steps. This variability could be attributed to the small scale of the experiment, constrained by limited computational resources. In the original paper (Figure 5 a), we can also see the accuracies of both attempts improve in twists and turns.


## License

[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Discord Server

Join our Discord server [here](https://discord.gg/xhcBDEM3).

## Feeling Generous? 😊

Eager to buy me a cup of $2 coffee or iced tea? 🍵☕ Sure, here is the link: [https://ko-fi.com/drnicefellow](https://ko-fi.com/drnicefellow). Please add a note on which one you want me to drink?