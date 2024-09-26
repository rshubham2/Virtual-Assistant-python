import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

def load_model(init_from='resume', out_dir='out', device='cuda', dtype=None):
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype = dtype or ('bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)

    return model, ctx

def load_encoder_decoder(init_from='resume', checkpoint=None, dataset_dir='data', gpt2_encoding='gpt2'):
    if init_from == 'resume' and checkpoint:
        meta_path = os.path.join(dataset_dir, checkpoint['config']['dataset'], 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
            return encode, decode
    enc = tiktoken.get_encoding(gpt2_encoding)
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    return encode, decode

def sample_from_model(model, ctx, start="\n", num_samples=1, max_new_tokens=100, temperature=0.8, top_k=200, encode=None, decode=None, device='cuda'):
    # If the start string begins with 'FILE:', read the content from the file
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()

    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    generated_texts = []  # Collect generated texts here

    # Generate and collect samples
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                generated_text = decode(y[0].tolist())
                generated_texts.append(generated_text)

    return generated_texts

# Example usage in another Python file (e.g., train_1.py):
# from sample import load_model, load_encoder_decoder, sample_from_model

# model, ctx = load_model(init_from='resume', out_dir='out', device='cuda')
# encode, decode = load_encoder_decoder(init_from='resume', checkpoint=checkpoint)
# generated_texts = sample_from_model(model, ctx, start="\n", num_samples=1, max_new_tokens=100, encode=encode, decode=decode)
# print(generated_texts)  # This will print the generated samples