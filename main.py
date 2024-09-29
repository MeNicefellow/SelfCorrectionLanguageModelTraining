# Install required libraries if not already installed
# You can uncomment the following lines to install them
# !pip install transformers datasets accelerate torch

import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse

from transformers import StoppingCriteria, StoppingCriteriaList


class SpecialTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, special_token_id):
        self.special_token_id = special_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated token is the special token
        return input_ids[0, -1] == self.special_token_id


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train Language Models to Self-Correct via Reinforcement Learning')
parser.add_argument('--model_name', type=str, default='unsloth/Llama-3.2-3B-Instruct', help='Name of the model to train')
parser.add_argument('--dataset_name', type=str, default='lighteval/MATH', help='Name of the dataset to use')
parser.add_argument('--question_column', type=str, default='problem', help='Column name for the questions in the dataset')
parser.add_argument('--answer_column', type=str, default='solution', help='Column name for the gold standard answers in the dataset')
args = parser.parse_args()

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open files for writing
accuracy_file = open('accuracies.txt', 'w')
generation_file = open('generations.txt', 'w')

# Load the model and tokenizer
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure the tokenizer has a pad token

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.train()  # Set the model to training mode

# Load the base (reference) model for KL divergence calculation
base_model = AutoModelForCausalLM.from_pretrained(model_name)
base_model.to(device)
base_model.eval()  # Set the base model to evaluation mode
for param in base_model.parameters():
    param.requires_grad = False  # Disable gradient computation for the base model

# Load the dataset
dataset = load_dataset(args.dataset_name)

stopping_criteria = StoppingCriteriaList([SpecialTokenStoppingCriteria(128009)])

# Define the reward function
def is_correct(model_to_use, problem, solution, attempt, generation_file):
    # Formulate the prompt
    prompt = f"For the math problem:\n{problem}\n\nThe correct answer is:\n{solution}\n\nIs the following solution correct?\n{attempt}\nPlease answer with Yes or No."

    messages = [
        {"role": "user", "content": prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize and generate response
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    model_to_use.eval()
    with torch.no_grad():
        outputs = model_to_use.generate(**inputs, max_length=1264, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    # Write prompt and response to generation_file
    generation_file.write("-=-=-=-=-=-=-=-=\nPrompt for checking correctness:\n" + prompt_text + '\n')
    generation_file.write("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nResponse:\n" + response + '\n')
    generation_file.flush()  # Ensure data is written to file

    return 'Yes' in response[:10]

# Updated reward shaping to align with SCoRe paper
def compute_reward(problem, solution, y1, y2, generation_file, current_stage):
    # Rewards for each attempt
    r1 = 1.0 if is_correct(model, problem, solution, y1, generation_file) else 0.0
    r2 = 1.0 if is_correct(model, problem, solution, y2, generation_file) else 0.0

    # Reward shaping bonus
    bonus = 0.0
    if current_stage == 2:
        bonus = alpha * (r2 - r1)

    return r1, r2, bonus

def generate_both_attempts(model_to_use, problem, generation_file):
    model_to_use.eval()
    with torch.no_grad():

        messages = [
            {"role": "user", "content": problem}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        outputs = model_to_use.generate(**inputs, max_length=2048, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria)
        attempt1 = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        messages.append({"role": "assistant", "content": attempt1})
        messages.append({"role": "user", "content": 'You are given a second chance to answer this. Please correct any error in your first answer and provide a corrected solution directly.'})
        correction_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(correction_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        outputs = model_to_use.generate(**inputs, max_length=2048, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria)
        attempt2 = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        # Write prompts and attempts to generation_file
        generation_file.write('--------------------------------\nPrompt for first attempt:\n' + prompt + '\n')
        generation_file.write("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nFirst attempt:\n" + attempt1 + '\n')
        generation_file.write('----------\nPrompt for second attempt:\n' + correction_prompt + '\n')
        generation_file.write("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nSecond attempt:\n" + attempt2 + '\n')
        generation_file.flush()  # Ensure data is written to file

    return attempt1, prompt, attempt2, correction_prompt

def evaluate_both_attempts(model_to_use, dataset_split, generation_file):
    model_to_use.eval()
    correct_attempts = 0
    correct_attempts2 = 0
    total_attempts = 0

    for i, batch in enumerate(tqdm(dataset_split)):
        if i >= 100:  # Limit to first 100 samples
            break

        problem = batch[args.question_column]
        solution = batch[args.answer_column]

        y1, prompt1, y2, prompt2 = generate_both_attempts(model_to_use, problem, generation_file)

        # Check correctness of the first attempt
        if is_correct(model_to_use, problem, solution, y1, generation_file):
            correct_attempts += 1
        # Check correctness of the second attempt
        if is_correct(model_to_use, problem, solution, y2, generation_file):
            correct_attempts2 += 1
        total_attempts += 1
    accuracy = correct_attempts / total_attempts if total_attempts > 0 else 0
    accuracy2 = correct_attempts2 / total_attempts if total_attempts > 0 else 0
    return accuracy, accuracy2

# Set hyperparameters
alpha = 0.5  # Reward shaping coefficient
beta_stage1 = 0.2  # KL divergence regularization coefficient for the first attempt in Stage I
beta1 = 0.1  # KL divergence regularization coefficient for the first attempt in Stage II
beta2 = 0.05  # KL divergence regularization coefficient for the second attempt in Stage II
num_epochs = 1  # Adjust as needed
learning_rate = 1e-5
evaluation_interval = 50  # Evaluate every 50 steps

# Set up the optimizer
optimizer = Adam(model.parameters(), lr=learning_rate)

# Evaluate the base model on the test split before training
accuracy_file.write("Evaluating the base model on the test split for the attempts...\n")
base_model_accuracy, base_model_accuracy2 = evaluate_both_attempts(base_model, dataset['test'], generation_file)
accuracy_file.write(f"First Attempt Base Model Test Accuracy: {base_model_accuracy * 100:.2f}%\n")
accuracy_file.write(f"Second Attempt Base Model Test Accuracy: {base_model_accuracy2 * 100:.2f}%\n\n")
accuracy_file.flush()

# Function to compute log probabilities of generated tokens
def compute_log_probabilities(outputs, labels):
    logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)  # Log probabilities

    # Shift the logits and labels to align them
    shift_log_probs = log_probs[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Flatten the tensors
    shift_log_probs = shift_log_probs.view(-1, shift_log_probs.size(-1))
    shift_labels = shift_labels.view(-1)

    # Select the log probabilities corresponding to the generated tokens
    selected_log_probs = shift_log_probs[torch.arange(shift_labels.size(0)), shift_labels]

    # Sum the log probabilities
    log_prob = selected_log_probs.sum()
    return log_prob

# Training Loop
global_step = 0
current_stage = 1  # Start with Stage I
for epoch in range(num_epochs):
    accuracy_file.write(f"Epoch {epoch + 1}/{num_epochs}\n")
    model.train()
    for batch in tqdm(dataset['train']):
        problem = batch[args.question_column]
        solution = batch[args.answer_column]

        # Generate first and second attempts
        y1, prompt1, y2, prompt2 = generate_both_attempts(model, problem, generation_file)

        # Compute rewards
        r1, r2, bonus = compute_reward(problem, solution, y1, y2, generation_file, current_stage)

        # Prepare inputs for KL divergence computation
        # First attempt
        full_text1 = prompt1 + y1
        inputs1 = tokenizer(full_text1, return_tensors="pt", truncation=True, max_length=1024).to(device)
        labels1 = inputs1.input_ids.clone()

        # Mask non-attempt tokens for first attempt
        prompt1_tokenized = tokenizer(prompt1, return_tensors="pt", truncation=True, max_length=1024).to(device)
        prompt_length1 = prompt1_tokenized.input_ids.shape[1]
        labels1[0][:prompt_length1] = -100  # Mask prompt tokens

        # Second attempt
        full_text2 = prompt2 + y2
        inputs2 = tokenizer(full_text2, return_tensors="pt", truncation=True, max_length=1024).to(device)
        labels2 = inputs2.input_ids.clone()

        # Mask non-attempt tokens for second attempt
        prompt2_tokenized = tokenizer(prompt2, return_tensors="pt", truncation=True, max_length=1024).to(device)
        prompt_length2 = prompt2_tokenized.input_ids.shape[1]
        labels2[0][:prompt_length2] = -100  # Mask prompt tokens

        # Compute log probabilities for first attempt
        model_outputs1 = model(**inputs1)
        log_prob_y1 = compute_log_probabilities(model_outputs1, labels1)

        # Compute KL divergence for first attempt
        with torch.no_grad():
            base_outputs1 = base_model(**inputs1)
        kl_div_y1 = F.kl_div(
            F.log_softmax(model_outputs1.logits, dim=-1),
            F.softmax(base_outputs1.logits, dim=-1),
            reduction='batchmean'
        )

        # Compute log probabilities for second attempt
        model_outputs2 = model(**inputs2)
        log_prob_y2 = compute_log_probabilities(model_outputs2, labels2)

        # Compute KL divergence for second attempt
        with torch.no_grad():
            base_outputs2 = base_model(**inputs2)
        kl_div_y2 = F.kl_div(
            F.log_softmax(model_outputs2.logits, dim=-1),
            F.softmax(base_outputs2.logits, dim=-1),
            reduction='batchmean'
        )

        # Compute total loss based on the current stage
        if current_stage == 1:
            # Stage I: Optimize second attempt, keep first attempt close to base model
            loss = - r2 * log_prob_y2 + beta_stage1 * kl_div_y1
        else:
            # Stage II: Optimize both attempts with reward shaping
            loss = - (r1 * log_prob_y1 + r2 * log_prob_y2 + bonus * (log_prob_y2 - log_prob_y1)) + beta1 * kl_div_y1 + beta2 * kl_div_y2

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        # Switch stages every 50 steps
        if global_step % 50 == 0:
            current_stage = 2 if current_stage == 1 else 1

            # Evaluate every 'evaluation_interval' steps
            accuracy_file.write(f"\nEvaluating at step {global_step}...\n")
            model_accuracy, model_accuracy2 = evaluate_both_attempts(model, dataset['test'], generation_file)
            accuracy_file.write(f"First Attempt Model Test Accuracy: {model_accuracy * 100:.2f}%\n")
            accuracy_file.write(f"Second Attempt Model Test Accuracy: {model_accuracy2 * 100:.2f}%\n\n")
            accuracy_file.flush()

    # Save the model after each epoch
    model.save_pretrained(f"finetuned_model_epoch_{epoch + 1}")
    tokenizer.save_pretrained(f"finetuned_model_epoch_{epoch + 1}")

# Final Evaluation
accuracy_file.write("Starting Final Evaluation on the test split...\n")
model_accuracy, model_accuracy2 = evaluate_both_attempts(model, dataset['test'], generation_file)
accuracy_file.write(f"First Attempt Model Test Accuracy: {model_accuracy * 100:.2f}%\n")
accuracy_file.write(f"Second Attempt Model Test Accuracy: {model_accuracy2 * 100:.2f}%\n\n")
accuracy_file.flush()

# Close the files
accuracy_file.close()
generation_file.close()
