# %%
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

subject_letters = "abcde"
random_subject_letter = lambda: random.choice(subject_letters)

filler_letters = "fghijklmnopqrstuvwxyz"
random_filler_letter = lambda: random.choice(filler_letters)


def get_ioi_sample():
    """
    Eg. Alice and Bob went to the shop, and Bob bought Alice a present.
    """
    alice = random_subject_letter()
    bob = random_subject_letter()

    return (
        alice
        + random_filler_letter()
        + bob
        + "".join([random_filler_letter() for _ in range(random.randint(3, 10))])
        + bob
        + random_filler_letter()
        + alice
    )


def get_induction_sample():
    alice = random_subject_letter()
    bob = random_subject_letter()

    return (
        alice
        + bob
        + "".join([random_filler_letter() for _ in range(random.randint(3, 10))])
        + alice
        + bob
    )


letters = "abcdefghijklmnopqrstuvwxyz"
random_letter = lambda: random.choice(letters)


def get_random_sample():
    return "".join([random_letter() for _ in range(random.randint(10, 40))])


class ToyDataset(Dataset):
    def __init__(
        self,
        num_samples=2000,
        max_length=20,
        include_random=True,
        include_induction=False,
        include_ioi=False,
    ):
        super().__init__()
        self.num_samples = num_samples

        sampling_functions = []
        if include_random:
            sampling_functions.append(get_random_sample)
        if include_induction:
            sampling_functions.append(get_induction_sample)
        if include_ioi:
            sampling_functions.append(get_ioi_sample)
        self.samples = [random.choice(sampling_functions)() for _ in range(num_samples)]

        # We’ll define a simple letter-to-index mapping
        self.char2idx = {c: i for i, c in enumerate(letters)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}

        self.max_length = max_length  # for optional padding if we want

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.samples[idx]
        # Convert each character into an integer token
        token_ids = [self.char2idx[ch] for ch in text]
        return torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch):
    """
    Collate function to pad variable-length sequences to the same length.
    PyTorch transformers typically expect [sequence_length, batch_size]
    or [batch_size, sequence_length]. We'll do [batch_size, seq_len].
    """
    # Find max length in this batch
    max_len = max(x.size(0) for x in batch)

    padded = []
    for x in batch:
        # Pad with -100 or 0 (depending on how you want your loss).
        # For cross-entropy, you can ignore index -100 or add special tokens.
        # Here, we'll just pad with 0 for simplicity (which corresponds to 'a').
        pad_size = max_len - x.size(0)
        padded_tokens = torch.cat([x, torch.zeros(pad_size, dtype=torch.long)])
        padded.append(padded_tokens)

    return torch.stack(padded, dim=0)  # [batch_size, max_seq_len]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x shape: [seq_len, batch_size, d_model]
        We'll add positional encoding up to seq_len.
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :]


class SimpleTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size=26,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # PyTorch’s default in nn.Transformer is seq_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        """
        Create a causal (subsequent) mask of shape [seq_len, seq_len],
        where positions can only attend to themselves and the past.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) == 1, diagonal=1)
        # True means we want to mask out (i.e. ignore) those positions
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, src, targets=None):
        """
        src: [batch_size, seq_len]
        targets: [batch_size, seq_len] (optional)
        """
        batch_size, seq_len = src.shape
        src = src.transpose(0, 1)  # => [seq_len, batch_size]

        x = self.embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)  # [seq_len, batch_size, d_model]

        mask = self.generate_causal_mask(seq_len, x.device)
        encoded = self.transformer_encoder(
            x, mask=mask
        )  # [seq_len, batch_size, d_model]
        logits = self.fc_out(encoded)  # [seq_len, batch_size, vocab_size]

        if targets is not None:
            targets = targets.transpose(0, 1)  # => [seq_len, batch_size]

            # We want:
            #   logits[i, :] to predict targets[i+1, :]
            # so let's ignore the last logit, and ignore the first target:

            # shift by 1
            logits = logits[:-1]  # [seq_len-1, batch_size, vocab_size]
            shifted_targets = targets[1:]  # [seq_len-1, batch_size]

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)), shifted_targets.reshape(-1)
            )
            return logits, loss
        else:
            return logits


# Create dataset and dataloader
dataset = ToyDataset(
    num_samples=20_000,
    include_random=False,
    include_induction=True,
    include_ioi=True,
)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)

# Instantiate model
model = SimpleTransformerModel(
    vocab_size=26, d_model=64, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)

        logits, loss = model(batch, targets=batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def sample_text(model, start_text, target, max_new_tokens=10):
    """
    Generate up to max_new_tokens after 'start_text'.
    """
    model.eval()
    with torch.no_grad():
        # Convert start_text to token IDs
        input_ids = (
            torch.tensor([dataset.char2idx[ch] for ch in start_text], dtype=torch.long)
            .unsqueeze(0)
            .to(device)
        )  # [1, length]

        for _ in range(max_new_tokens):
            # We only pass in the entire sequence each time;
            # the model sees it and we sample next token from final position's logits
            logits = model(
                input_ids
            ).detach()  # shape [seq_len, batch_size, vocab_size]
            # Take the last position: logits[-1, 0, :]
            next_token_logits = logits[-1, 0, :]
            next_token_id = torch.argmax(next_token_logits)  # greedy
            # Append to input
            input_ids = torch.cat(
                [input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1
            )

        # Convert back to string
        output_tokens = input_ids[0].cpu().tolist()
        output_text = "".join([dataset.idx2char[t] for t in output_tokens])
        return output_text, output_text[-1] == target


# Evaluate 
induction_answers = []
for _ in range(100):
    start_text = get_induction_sample()
    output_text, correct = sample_text(
        model, start_text[:-1], start_text[-1], max_new_tokens=1
    )
    induction_answers.append(correct)
print(f"Induction accuracy: {sum(induction_answers)} / 100")

ioi_answers = []
for _ in range(100):
    start_text = get_ioi_sample()
    output_text, correct = sample_text(model, start_text[:-1], start_text[-1], max_new_tokens=1)
    ioi_answers.append(correct)
print(f"IOI accuracy: {sum(ioi_answers)} / 100")

# %%