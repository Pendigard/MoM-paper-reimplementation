from lm_eval.api.model import LM
from lm_eval import simple_evaluate
from transformers import AutoTokenizer, GPT2LMHeadModel
import os
import sys
import torch
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.module.naive_mom import MoM, LinearAttention, GLAAttention, GDeltaAttention
from src.module.mom_llm import MoMLLM

class LocalModelWrapper(LM):
    def __init__(self, model, tokenizer, device="cuda"):
        super().__init__()
        
        # Use your custom model and tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        # Set padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Custom model loaded successfully on {device}")
    
    def loglikelihood(self, requests):
        """Calculate log-likelihood for each request"""
        results = []
        
        for request in requests:
            context, continuation = request.args

            # Tokenize context and continuation
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
            continuation_tokens = self.tokenizer.encode(continuation, add_special_tokens=False)
            full_tokens = context_tokens + continuation_tokens
            
            # Get model outputs
            with torch.no_grad():
                input_ids = torch.tensor([full_tokens]).to(self.device)
                
                # MODEL FORWARD PASS
                # Adjust this based on your model's output format
                outputs = self.model(input_ids)

                # If model returns a tuple/dict, extract logits
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # or outputs.logits if named tuple
                elif isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
            
            # Calculate log-likelihood for continuation tokens
            logprobs = F.log_softmax(logits[0], dim=-1)
            continuation_logprobs = []
            
            for i, token_id in enumerate(continuation_tokens):
                pos = len(context_tokens) + i - 1
                if pos >= 0 and pos < len(logprobs):
                    continuation_logprobs.append(logprobs[pos, token_id].item())
            
            total_logprob = sum(continuation_logprobs)
            is_greedy = True
            
            results.append((total_logprob, is_greedy))
        
        return results
    
    def generate_until(self, requests):
        """Generate text until stopping condition"""
        results = []
        
        for i, request in enumerate(requests):
            context = request.args[0]
            generation_kwargs = request.args[1]
            
            # Tokenize input
            input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
            
            # Generate tokens autoregressively
            max_new_tokens = generation_kwargs.get("max_gen_toks", 256)
            generated_ids = input_ids.clone()
            
            with torch.no_grad():
                for i in range(max_new_tokens):
                    print(f"\t\tProcessed {i} tokens.")
                    # YOUR MODEL FORWARD PASS
                    outputs = self.model(generated_ids)
                    
                    # Extract logits based on your model's output format
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    elif isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    # Get next token (greedy decoding)
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Append to generated sequence
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    # Check for stopping conditions
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Check for until strings if specified
                    until = generation_kwargs.get("until", [])
                    if until:
                        current_text = self.tokenizer.decode(
                            generated_ids[0][input_ids.shape[1]:],
                            skip_special_tokens=True
                        )
                        if any(stop_str in current_text for stop_str in until):
                            break
            
            # Decode only the generated part
            generated_text = self.tokenizer.decode(
                generated_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            results.append(generated_text)
        
        return results
    
    def loglikelihood_rolling(self, requests):
        """Calculate rolling log-likelihood (for perplexity)"""
        return self.loglikelihood(requests)

CONFIG = {
    "vocab_size": 32000,
    "dim": 256,
    "num_layers": 4,
    "num_memories": 4,
    "hidden_dim": 256,
    "top_k": 2,
    "seq_len": 512,
    "batch_size": 4,
    "lr": 3e-4,
    "max_steps": 50000
}

LIMIT = 1
BATCH_SIZE = 1
TASKS = ["hellaswag"]
HF = False

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not HF:
        # load model class, checkpoint and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        update_module = GLAAttention(
            input_dim=CONFIG["dim"],
            hidden_dim=CONFIG["hidden_dim"],
            num_memories=CONFIG["num_memories"]
        )
        model = MoMLLM(
            vocab_size=CONFIG["vocab_size"],
            hidden_dim=CONFIG["hidden_dim"],
            num_memories=CONFIG["num_memories"],
            k=CONFIG["top_k"],
            num_layers=CONFIG["num_layers"],
            update_module=update_module
        )

        dir_path = "/Vrac/21400184/Projet_deepl/"
        cpt_path = "mom_gla__final_slimpajama_step50000.pt"
        path = dir_path + cpt_path

        if os.path.exists(path):
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f"Failed to find file: {path}")
            sys.exit(0)
        model.to(device)
        model.eval()
        
        # Create wrapper
        model_wrapper = LocalModelWrapper(
            model=model,
            tokenizer=tokenizer,
            device=device
        )
    else:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        model_wrapper = LocalModelWrapper(
            model=model,
            tokenizer=tokenizer,
            device=device
        )

    # Run evaluation
    print("\nRunning evaluation...")
    results = simple_evaluate(
        model=model_wrapper,
        tasks=TASKS,
        num_fewshot=0,
        batch_size=BATCH_SIZE,
        limit=LIMIT
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for task, metrics in results['results'].items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")