import torch
from torch.utils.data import Dataset

class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        PyTorch Dataset for ticket classification.
        """
        self.texts = texts.reset_index(drop=True) if hasattr(texts, 'reset_index') else texts
        self.labels = labels.reset_index(drop=True) if hasattr(labels, 'reset_index') else labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Use direct tokenizer call instead of encode_plus
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }