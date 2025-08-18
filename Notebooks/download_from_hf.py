import gpt
import torch
import tiktoken
from gpt import generate, text_to_token_ids, token_ids_to_text

BASE_CONFIG = {
    "vocab_size": 50257,   
    "context_length": 1024, 
    "drop_rate": 0.0,       
    "qkv_bias": True,
    'emb_dim':768,
    'n_layers':12,
    'n_heads':12        
}


file_name = "gpt2-small-124M.pth"

model = gpt.GPT(BASE_CONFIG)
model.load_state_dict(torch.load(file_name, weights_only=True))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device);

tokenizer = tiktoken.get_encoding("gpt2")

def get_response(text):

    token_ids = generate(
        model=model.to(device),
        idx=text_to_token_ids(text, tokenizer).to(device),
        max_new_tokens=30,
        context_size=BASE_CONFIG["context_length"],
        top_k=1,
        temperature=1.0
    )
    print(token_ids)
    print("AI:\n", token_ids_to_text(token_ids, tokenizer))

print("Ask your query:\n")
while True:
    text = input()
    if text == 'exit':
        break
    get_response(text)