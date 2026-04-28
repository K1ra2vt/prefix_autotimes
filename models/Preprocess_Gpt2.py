import torch
import torch.nn as nn
from transformers import (
    GPT2Model,
    GPT2Tokenizer,
)

class Gpt2Model(nn.Module):
    def __init__(self, configs):
        super(Gpt2Model, self).__init__()
        self.device = configs.gpu
        print(self.device)
        
        self.gpt2 = GPT2Model.from_pretrained(configs.llm_ckp_dir, local_files_only=True).to(self.device)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_ckp_dir, local_files_only=True)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.vocab_size = self.gpt2_tokenizer.vocab_size
        self.hidden_dim_of_gpt2 = 768
        
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

    def tokenizer(self, x):
        output = self.gpt2_tokenizer(x, return_tensors="pt")['input_ids'].to(self.device)
        result = self.gpt2.get_input_embeddings()(output)
        return result   
    
    def forecast(self, x_mark_enc, context_features=None):
        # x_mark_enc: [bs x T x hidden_dim_of_gpt2] or list of strings
        # context_features: dict of context features for each sample

        # Handle text tokenization
        if isinstance(x_mark_enc, list):
            # x_mark_enc is a list of strings (from Dataset_Preprocess)
            x_mark_enc = torch.cat([self.tokenizer([text]) for text in x_mark_enc], 0)
        else:
            # x_mark_enc is tensor embeddings
            x_mark_enc = torch.cat([self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))], 0)

        text_outputs = self.gpt2(inputs_embeds=x_mark_enc).last_hidden_state
        text_outputs = text_outputs[:, -1, :]  # [bs, hidden_dim]

        # Process context features if provided
        if context_features is not None:
            # context_features is a list of dicts, one per sample
            feature_vectors = []
            for sample_features in context_features:
                # Concatenate all features for this sample into a single vector
                sample_vector = []
                for key in sorted(sample_features.keys()):  # Sort for consistent ordering
                    feature_tensor = sample_features[key]
                    if feature_tensor.dim() > 0:
                        # If tensor has multiple values, take mean or flatten
                        sample_vector.append(torch.mean(feature_tensor).unsqueeze(0))
                    else:
                        sample_vector.append(feature_tensor.unsqueeze(0))
                if sample_vector:
                    feature_vectors.append(torch.cat(sample_vector, dim=0))
                else:
                    # If no features, use zero vector
                    feature_vectors.append(torch.zeros(1, dtype=torch.float32).to(self.device))

            context_outputs = torch.stack(feature_vectors, dim=0).to(self.device)  # [bs, num_features]

            # Concatenate text embeddings with context features
            combined_outputs = torch.cat([text_outputs, context_outputs], dim=-1)
            return combined_outputs
        else:
            return text_outputs
    
    def forward(self, x_mark_enc, context_features=None):
        return self.forecast(x_mark_enc, context_features)