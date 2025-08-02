"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

3) https://github.com/karpathy/nanoGPT/blob/master/
"""

from .gpt import GPT, GPTConfig
import os
import pickle
import torch

__all__=['nanoGPT']

# config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('/home/pandaa2/AIRC/IBM_AIRC_LLM/CompOptim/config/train_shakespeare_char.py').read()) # overrides from your config file
exec(open('/home/pandaa2/AIRC/IBM_AIRC_LLM/CompOptim/config/train_gpt2.py').read())
# config = {k: globals()[k] for k in config_keys} # will be useful for logging

# attempt to derive vocab_size from the dataset
data_dir = '/home/pandaa2/AIRC/IBM_AIRC_LLM/CompOptim/data/OpenWebText' # to be changed everytime if used a different setting
# data_dir = '/home/pandaa2/AIRC/IBM_AIRC_LLM/CompOptim/data/Shakespeare'
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

# Check if init_from is defined in the config file
init_from = globals().get('init_from', 'scratch')

if init_from == 'scratch':
    # print("Initializing a new model from scratch")
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
        dropout=dropout
    )
    gptconf = GPTConfig(**model_args)
elif init_from.startswith('gpt2'):
    # initialize from OpenAI GPT-2 weights
    model_args = {}
    override_args = dict(dropout=dropout)
    temp_model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(temp_model.config, k)
    gptconf = GPTConfig(**model_args)
elif init_from == 'resume':
    temp = "/home/pandaa2/AIRC/IBM_AIRC_LLM/CompOptim/results/"+globals().get('out_dir')
    ckpt_path = os.path.join(temp, "chk_Shakespeare-NewAlg-topk50-Rank-4-LR_Decay-cosine-LR-0.0001-Variety-index-BS-128-K-50-10_Epoch_10.pt") # add the checkpoint file name here
    checkpoint = torch.load(ckpt_path)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    model_args = {}
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size',  'vocab_size']:#'bias',
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
else:
    raise ValueError(f"Unknown init_from: {init_from}. Expected 'scratch' or 'gpt2-*'")

new_out_dir = "/home/pandaa2/AIRC/IBM_AIRC_LLM/CompOptim/results/"+globals().get('out_dir')

class nanoGPT(GPT):
    def __init__(self):
        super().__init__(gptconf)
        self.config = gptconf
        self.out_dir = new_out_dir
        if block_size < self.config.block_size:
            self.crop_block_size(block_size)
            model_args['block_size'] = block_size
        self.model_args = model_args
        
        if init_from == 'resume':
            state_dict = checkpoint['model']
            super().load_state_dict(state_dict)
            self.iter_num = checkpoint['iter_num']
            self.best_val_loss = checkpoint['best_val_loss']
