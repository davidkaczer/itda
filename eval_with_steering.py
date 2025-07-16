"""Usage:
python eval_with_steering.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml --steering_config steering_config.yaml
"""

import asyncio
import logging
import os
import yaml
import json
import torch
from transformers import AutoTokenizer
import pandas as pd
import random
from typing import Optional, List, Dict, Any
from transformer_lens import HookedTransformer
import numpy as np

from judge import OpenAiJudge


def sample_with_steering(
    model: HookedTransformer,
    conversations: List[List[Dict]],
    steering_vectors: Optional[List[Dict]] = None,
    top_p=1,
    max_tokens=600,
    temperature=1,
    stop=[],
    min_tokens=1,
):
    """Sample from model with optional steering vectors applied using batching."""
    
    def steering_hook(activations, hook, vector, coeff, layer_idx):
        """Hook function to add steering vector to activations."""
        if len(activations.shape) == 3:  # [batch, seq, hidden]
            # Add steering vector to the last position for all items in batch
            # Vector is broadcasted across batch dimension
            activations[:, -1, :] += coeff * vector.unsqueeze(0)
        return activations
    
    # Convert all conversations to text
    texts = []
    for messages in conversations:
        if len(messages) == 1 and messages[0]["role"] == "user":
            text = messages[0]["content"]
        else:
            # Handle multi-turn conversations if needed
            text = messages[-1]["content"]  # Use last user message for simplicity
        texts.append(text)
    
    # Batch tokenize all texts
    all_tokens = []
    original_lengths = []
    max_length = 0
    
    for text in texts:
        tokens = model.to_tokens(text, prepend_bos=True)
        all_tokens.append(tokens)
        original_lengths.append(tokens.shape[1])
        max_length = max(max_length, tokens.shape[1])
    
    # Pad all token sequences to the same length
    # Use the tokenizer's pad_token_id, or 0 if not available
    pad_token_id = getattr(model.tokenizer, 'pad_token_id', 0)
    if pad_token_id is None:
        pad_token_id = 0
    
    batch_tokens = torch.full(
        (len(all_tokens), max_length), 
        pad_token_id,
        dtype=torch.long, 
        device=model.cfg.device
    )
    
    # Create attention mask for proper padding handling
    attention_mask = torch.zeros(
        (len(all_tokens), max_length),
        dtype=torch.bool,
        device=model.cfg.device
    )
    
    for i, tokens in enumerate(all_tokens):
        seq_len = tokens.shape[1]
        batch_tokens[i, :seq_len] = tokens[0]  # Remove singleton batch dim
        attention_mask[i, :seq_len] = True  # Mark real tokens
    
    # Setup hooks for steering if provided
    hooks = []
    if steering_vectors:
        for steering in steering_vectors:
            layer_idx = steering["layer"]
            vector = steering["vector"].to(model.cfg.device)
            coeff = steering["coefficient"]
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            
            def make_hook(vec, c, idx):
                return lambda activations, hook: steering_hook(activations, hook, vec, c, idx)
            
            hooks.append((hook_name, make_hook(vector, coeff, layer_idx)))
    
    # Custom generation with proper EOS handling for batches
    def generate_batch_with_eos_handling():
        batch_size = batch_tokens.shape[0]
        current_tokens = batch_tokens.clone()
        eos_token_id = model.tokenizer.eos_token_id
        
        # Track which sequences are still generating
        active_sequences = torch.ones(batch_size, dtype=torch.bool, device=model.cfg.device)
        
        for _ in range(max_tokens):
            if not active_sequences.any():
                break
                
            # Get logits for current tokens
            logits = model(current_tokens)[:, -1, :]  # [batch, vocab]
            
            # Apply temperature and top_p sampling
            if temperature != 1.0:
                logits = logits / temperature
                
            if top_p < 1.0:
                # Apply top-p filtering
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Create mask for tokens to keep
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample next tokens
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch]
            
            # Only update active sequences
            next_tokens = next_tokens * active_sequences + pad_token_id * (~active_sequences)
            
            # Append to current tokens
            current_tokens = torch.cat([current_tokens, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update active sequences (stop if EOS generated)
            if eos_token_id is not None:
                active_sequences = active_sequences & (next_tokens != eos_token_id)
        
        return current_tokens
    
    # Generate with steering using context manager
    if hooks:
        # Add hooks to model
        for hook_name, hook_fn in hooks:
            model.add_hook(hook_name, hook_fn)
        
        try:
            batch_output = generate_batch_with_eos_handling()
        finally:
            # Remove hooks after generation
            model.reset_hooks()
    else:
        batch_output = generate_batch_with_eos_handling()
    
    # Extract generated text for each item in batch
    answers = []
    for i, original_length in enumerate(original_lengths):
        generated_tokens = batch_output[i, original_length:]
        
        # Remove padding tokens and stop at EOS if present
        generated_tokens_clean = []
        eos_token_id = model.tokenizer.eos_token_id
        
        for token in generated_tokens:
            if token.item() == pad_token_id:
                continue  # Skip padding tokens
            if eos_token_id is not None and token.item() == eos_token_id:
                break  # Stop at EOS token
            generated_tokens_clean.append(token.item())
        
        if generated_tokens_clean:
            generated_text = model.to_string(torch.tensor(generated_tokens_clean))
        else:
            generated_text = ""
            
        answers.append(generated_text)
    
    return answers


def create_steering_vectors(model: HookedTransformer, steering_config: Dict[str, Any]) -> List[Dict]:
    """Create steering vectors based on configuration."""
    steering_vectors = []
    
    for config in steering_config.get("vectors", []):
        prompt_pos = config["positive_prompt"]
        prompt_neg = config.get("negative_prompt", "")
        layer = config["layer"]
        coefficient = config["coefficient"]
        
        # Get activations for positive prompt
        tokens_pos = model.to_tokens(prompt_pos, prepend_bos=True)
        _, cache_pos = model.run_with_cache(tokens_pos)
        
        # Get activation at specified layer (use mean across sequence)
        activation_pos = cache_pos[f"blocks.{layer}.hook_resid_post"].mean(dim=1)  # [batch, hidden]
        
        if prompt_neg:
            # Get activations for negative prompt and subtract
            tokens_neg = model.to_tokens(prompt_neg, prepend_bos=True)
            _, cache_neg = model.run_with_cache(tokens_neg)
            activation_neg = cache_neg[f"blocks.{layer}.hook_resid_post"].mean(dim=1)
            
            steering_vector = activation_pos - activation_neg
        else:
            steering_vector = activation_pos
        
        # Normalize if requested
        if config.get("normalize", False):
            steering_vector = steering_vector / torch.norm(steering_vector)
        
        steering_vectors.append({
            "vector": steering_vector.squeeze(0),  # Remove batch dimension
            "layer": layer,
            "coefficient": coefficient,
            "name": config.get("name", f"steering_{layer}")
        })
    
    return steering_vectors


def load_model_for_steering(model_name: str, device: str = "auto") -> HookedTransformer:
    """Load model using HookedTransformer for steering capabilities."""
    print(f"Loading model {model_name} with HookedTransformer...")
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model with HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    return model


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: Optional[list[str]] = None,
        judge_prompts: Optional[dict] = None,
        temperature: float = 1,
        system: str = None,
        judge: str = "gpt-4o",
        **ignored_extra_args,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {
            metric: OpenAiJudge(judge, prompt)
            for metric, prompt in judge_prompts.items()
        } if judge_prompts else {}

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role="user", content=i)] for i in paraphrases]

    async def eval(self, model, n_per_question, steering_vectors=None):
        paraphrases, conversations = self.get_input(n_per_question)
        answers = sample_with_steering(
            model, 
            conversations, 
            steering_vectors=steering_vectors,
            temperature=self.temperature
        )
        
        df = pd.DataFrame(
            [
                dict(question=question, answer=answer, question_id=self.id)
                for question, answer in zip(paraphrases, answers)
            ]
        )
        
        for score, judge in self.judges.items():
            scores = await asyncio.gather(
                *[
                    judge(question=question, answer=answer)
                    for question, answer in zip(paraphrases, answers)
                ]
            )
            df[score] = scores
        return df


def load_questions(path):
    questions = []
    with open(path, "r") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f]
            is_json = True
        else:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            is_json = False
    
    for i, question in enumerate(data):
        if is_json:
            questions.append(
                Question(
                    id=f"{path}_{i}",
                    paraphrases=[question["messages"][0]["content"]],
                    judge_prompts={},
                )
            )
        else:
            questions.append(Question(**question))
    return questions


def load_steering_config(path: str) -> Dict[str, Any]:
    """Load steering configuration from YAML file."""
    if not path:
        return {}
    
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def main(
    model_name: str,
    questions_path: str,
    n_per_question: int = 10,
    output: str = "eval_result_steering.csv",
    steering_config_path: Optional[str] = None,
):
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    model = load_model_for_steering(model_name)
    
    # Load steering configuration
    steering_config = load_steering_config(steering_config_path) if steering_config_path else {}
    
    # Create steering vectors if config provided
    steering_vectors = None
    if steering_config:
        print("Creating steering vectors...")
        steering_vectors = create_steering_vectors(model, steering_config)
        print(f"Created {len(steering_vectors)} steering vectors")
    
    # Load questions
    questions = load_questions(questions_path)
    
    # Check for existing progress
    processed_questions = set()
    if os.path.exists(output):
        try:
            existing_df = pd.read_csv(output)
            if not existing_df.empty:
                processed_questions = set(existing_df["question_id"].unique())
                print(f"Found existing progress with {len(processed_questions)} questions already processed")
        except Exception as e:
            print(f"Error reading existing output file: {e}")
    
    # Process questions
    all_results = []
    for question in questions:
        if question.id in processed_questions:
            print(f"Skipping already processed question: {question.id}")
            continue
        
        print(f"Processing question: {question.id}")
        question_result = asyncio.run(
            question.eval(model, int(n_per_question), steering_vectors)
        )
        all_results.append(question_result)
        
        # Save progress after each question
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_df.to_csv(output, index=False)
            print(f"Saved progress to {output}")


if __name__ == "__main__":
    import fire
    fire.Fire(main) 