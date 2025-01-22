#!/usr/bin/env python

import os
import math
import random
import argparse
import wandb
import torch
import logging
from tqdm import tqdm
from typing import Dict, List, Any, Iterator
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    GPT2TokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator

##############################################################################
# Configuration
##############################################################################


@dataclass
class TrainingConfig:
    project_name: str = "example_saes_llms"
    run_name_prefix: str = "small-pythia"
    model_output_dir: str = "./small_pythia_streaming_ckpts"

    # We no longer rely on multiple runs in this file.
    # Instead, a single seed is passed in via command line.

    max_train_steps: int = 500_000
    logging_steps: int = 50
    save_steps: int = 500
    warmup_steps: int = 200
    lr: float = 1e-4
    batch_size: int = 32
    block_size: int = 256
    buffer_size: int = 10000


##############################################################################
# Utilities
##############################################################################


class PileChunkedDataset(torch.utils.data.IterableDataset):
    """
    A streaming IterableDataset that:
      - Loads The Pile samples (or another dataset) via HF streaming
      - Shuffles within a buffer
      - Tokenizes on the fly
      - Streams out chunked blocks of length `block_size`

    This dataset yields (input_ids, attention_mask) pairs, each shape [block_size].
    """

    def __init__(
        self,
        tokenizer: GPT2TokenizerFast,
        split: str = "train",
        block_size: int = 256,
        buffer_size: int = 10000,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.seed = seed

        # Load streaming dataset
        self.raw_dataset = load_dataset(
            "SkyLion007/openwebtext",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        # Shuffle in streaming mode (requires a buffer).
        self.raw_dataset = self.raw_dataset.shuffle(buffer_size=buffer_size, seed=seed)

        self.text_buffer = []
        self.token_buffer = []

    def process_example(self, text: str):
        """
        Tokenize a text string into input_ids (no chunking here).
        """
        return self.tokenizer(
            text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]

    def chunk_tokens(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yields chunked blocks from self.token_buffer until we don't have enough
        tokens left to form another full block.
        """
        while len(self.token_buffer) >= self.block_size:
            chunk = self.token_buffer[: self.block_size]
            self.token_buffer = self.token_buffer[self.block_size :]
            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "attention_mask": torch.ones(self.block_size, dtype=torch.long),
            }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Main iterator. Streams from the raw dataset, accumulates tokens,
        and yields chunked samples.
        """
        for sample in self.raw_dataset:
            text = sample.get("text", "")
            token_ids = self.process_example(text)
            self.token_buffer.extend(token_ids)

            # Yield out all possible full blocks
            for chunk in self.chunk_tokens():
                yield chunk

        # If we exhaust the stream, yield final partial chunks if enough tokens remain
        while len(self.token_buffer) >= self.block_size:
            chunk = self.token_buffer[: self.block_size]
            self.token_buffer = self.token_buffer[self.block_size :]
            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "attention_mask": torch.ones(self.block_size, dtype=torch.long),
            }


def create_small_pythia_config() -> GPTNeoXConfig:
    """
    Create a small GPTNeoX config.
    Approx ~14M parameters, akin to smallest Pythia variant.
    """
    hidden_size = 128
    return GPTNeoXConfig(
        vocab_size=50257,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=6,
        num_attention_heads=4,
        max_position_embeddings=128,
        rotary_pct=1.0,
        rotary_emb_base=10000,
        bos_token_id=50256,
        eos_token_id=50256,
    )


##############################################################################
# Training Loop
##############################################################################


def train_one_run(cfg: TrainingConfig, seed: int, accelerator: Accelerator):
    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize Weights & Biases
    wandb.init(
        project=cfg.project_name,
        name=f"{cfg.run_name_prefix}-seed_{seed}",
        config={
            "seed": seed,
            "max_steps": cfg.max_train_steps,
            "block_size": cfg.block_size,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
        },
    )

    # Create tokenizer & model
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model_config = create_small_pythia_config()
    model = GPTNeoXForCausalLM(model_config)

    # (Optional) If you need a padding token:
    # tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    # model.resize_token_embeddings(len(tokenizer))

    # Create dataset
    train_dataset = PileChunkedDataset(
        tokenizer=tokenizer,
        split="train",
        block_size=cfg.block_size,
        buffer_size=cfg.buffer_size,
        seed=seed,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size)

    # Prepare optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_train_steps,
    )

    # Prepare with accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop
    global_step = 0
    model.train()
    for batch in tqdm(train_loader, disable=not accelerator.is_local_main_process):
        global_step += 1

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if (global_step % cfg.logging_steps) == 0:
            wandb.log({"train/loss": loss.item(), "train/step": global_step})

        if (global_step % cfg.save_steps) == 0:
            if accelerator.is_main_process:
                # Save model checkpoint
                ckpt_dir = os.path.join(
                    cfg.model_output_dir, f"seed_{seed}", f"step_{global_step}"
                )
                os.makedirs(ckpt_dir, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

        if global_step >= cfg.max_train_steps:
            break

    # Final save
    if accelerator.is_main_process:
        final_dir = os.path.join(cfg.model_output_dir, f"seed_{seed}", "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        # Log final model as artifact
        artifact = wandb.Artifact(
            name=f"{cfg.run_name_prefix}-seed_{seed}-model", type="model"
        )
        artifact.add_dir(final_dir)
        wandb.log_artifact(artifact)

    wandb.finish()


def main():
    logging.basicConfig(level=logging.INFO)

    # Command-line arg for seed
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for training"
    )
    args = parser.parse_args()

    # Create the config
    cfg = TrainingConfig()

    # Create accelerator
    accelerator = Accelerator()

    # Train exactly once for this seed
    train_one_run(cfg, args.seed, accelerator)

    if accelerator.is_main_process:
        print(f"Training run for seed={args.seed} finished.")


if __name__ == "__main__":
    main()
