import pandas as pd
import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import anndata as an
import scanpy as sc
import scipy
from transformers import BertConfig, BertForMaskedLM, AdamW, DataCollatorForLanguageModeling
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, TensorDataset


class BasicBert:
    """
    A basic BERT model class for masked language modeling.

    This class provides a simplified interface for initializing and working with
    a BERT model for masked language modeling tasks.
    """

    def __init__(self, vocab_size, max_length, mask_token=199, unk_token=198):
        """
        Initializes a new BasicBert object.

        Args:
            vocab_size (int): The size of the vocabulary.
            max_length (int): The maximum sequence length the model can handle.
            mask_token (int, optional): The ID of the mask token used in masked language modeling 
                                       (default: -1).
            unk_token (int, optional): The ID of the unknown token used for out-of-vocabulary words
                                      (default: -2).
        """
        self.model = None  
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.mask_token = mask_token
        self.unk_token = unk_token
        
        
    def build(
        self,
        num_hidden_layers=2,
        num_attention_heads=2,
        output_shape=10,
        **kwargs  
    ):
        """
        Builds and initializes the underlying BERT model.

        This method constructs a BERT model using the provided configuration parameters
        and stores it as the `model` attribute of the `BasicBert` object.

        Args:
            num_hidden_layers: The number of hidden layers in the transformer.
            num_attention_heads: The number of attention heads in each transformer layer.
            output_shape: The output dimension of the model (hidden size).
            **kwargs: Additional keyword arguments to pass to the `BertConfig` constructor. 
                      Refer to the `transformers` library documentation for valid options.
        """

        # Model Configuration
        config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=output_shape,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=output_shape * 2,
            max_position_embeddings=self.max_length,
            output_hidden_states=True,
            **kwargs
        )

        self.model = BertForMaskedLM(config)
        print('Model Built!')
        
        
    def _apply_masking(
        self,
        batch: torch.Tensor, 
        mlm_probability: float = 0.15
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies masked language modeling to a batch of input sequences.

        This function masks a random fraction of tokens in the input batch
        and creates corresponding labels for masked language modeling.

        Args:
            batch: The input sequences (token IDs).
            mask_token_id: The ID of the mask token.
            mlm_probability: The probability of masking a token (default 0.15).

        Returns:
            A tuple of:
                - The masked input batch (with some tokens replaced by mask_token_id).
                - The labels tensor (-100 for unmasked tokens, original token ID for masked tokens).

        """

        labels = batch.clone()   
        probability_matrix = torch.full(labels.shape, mlm_probability)

        # Special tokens mask (avoid masking special tokens like [CLS] and [SEP])
        special_tokens_mask = torch.zeros(labels.shape, dtype=torch.bool)
        special_tokens_mask[:, 0] = 1  # Mask first token ([CLS])
        special_tokens_mask[:, -1] = 1  # Mask last token ([SEP])

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Unmasked tokens get -100 label for ignoring during loss calculation

        batch[masked_indices] = self.mask_token
        return batch, labels
    
    
    def train(self, data_loader: DataLoader, num_epochs: int = 1, learning_rate: float = 1e-6, verbose: bool = True):
        """
        Trains the BERT model using the provided data.

        Args:
            data_loader: A DataLoader that provides batches of tokenized input data.
            num_epochs: The number of training epochs (default: 1).
            learning_rate: The learning rate for the AdamW optimizer (default: 1e-4).
            verbose: Whether to print training progress information (default: True).
        """

        if self.model is None:
            raise ValueError("BERT model not built. Call the 'build' method first.")

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, no_deprecation_warning=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch in data_loader:
                optimizer.zero_grad()
                
                # Apply masking and handle OOV tokens
                input_ids = batch[0].to(device)
                input_ids[input_ids >= self.vocab_size - 1] = self.unk_token
                masked_input_ids, labels = self._apply_masking(input_ids)

                # Forward and backward passes
                outputs = self.model(masked_input_ids, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss:.2f}")
                
        print('Done Training!')
        
    
    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Generates embeddings for a given input sequence using the BERT model.

        This method takes a tensor of token IDs as input, processes it through the BERT model,
        and returns the final hidden state embeddings, which are considered to be rich
        representations of the input text.

        Args:
            input_ids (torch.Tensor): A tensor containing the token IDs of the input sequence.

        Returns:
            torch.Tensor: The final hidden state embeddings of the input sequence.

        Raises:
            ValueError: If the BERT model has not been built yet (the `build` method needs to be
                        called before using this method).
        """

        if self.model is None:
            raise ValueError("BERT model not built. Call the 'build' method first.")

        # Ensure input is on the same device as the model (typically GPU if available)
        input_ids = input_ids.to(self.model.device) 

        # Get model's prediction (logits for each position)
        with torch.no_grad():  # No need to track gradients for inference
            outputs = self.model(input_ids)

        # Extract the final hidden state embeddings
        embeddings = outputs.hidden_states[-1]

        return embeddings

    




def format_input(token_list, max_length=12, pad_token=0):
    """
    Formats a list of tokens to a fixed length by truncating or padding.

    Args:
        token_list: The list of tokens to format.
        max_length: The desired fixed length.
        pad_token: The token ID used for padding (default is 0).

    Returns:
        The formatted list of tokens with length `max_length`.
    """
    
    # Truncate if longer than max_length
    token_list = list(token_list[:max_length]  )

    # Pad if shorter than max_length
    if len(token_list) < max_length:
        token_list.extend([pad_token] * (max_length - len(token_list)))

    return token_list



def prepare_bert_dataloader(tokens, batch_size=8, verbose=True):
    """Prepares a PyTorch DataLoader for BERT input from tokenized data.

    Args:
        tokens: A dictionary-like object containing 'input_ids' representing tokenized data.
        batch_size: The batch size to use in the DataLoader.
        verbose: Whether to print configuration and progress information.

    Returns:
        DataLoader: A PyTorch DataLoader object for iterating through the data in batches.
    """

    input_ids = torch.tensor(tokens['input_ids'].to_list())

    if verbose:  # Conditional printing
        print(f"Shape of input_ids: {input_ids.shape}")
        print(f"Data type of input_ids: {input_ids.dtype}")
        print(f"Device of input_ids: {input_ids.device}")
        print(f"First 10 values in input_ids: {input_ids[:10]}")

    dataset = TensorDataset(input_ids)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if verbose:
        print(f"Length of dataset: {len(dataset)}")
        print(f"Batch size: {data_loader.batch_size}")
        print(f"Number of batches: {len(data_loader)}")

        # Check the first batch
        for batch in data_loader:
            print(f"Shape of batch: {batch[0].shape}")
            print(f"Data type of batch: {batch[0].dtype}")
            print(f"Device of batch: {batch[0].device}")
            print(f"First 5 values in the first batch: {batch[0][0][:5]}")
            break

    print("done!")  # Always print the completion message, even if not verbose
    return data_loader


