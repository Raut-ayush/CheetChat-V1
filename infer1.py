from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TorchChatbot:
    def __init__(self, model_dir):
        print(f"🧠 Loading model from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.chat_history_ids = None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, user_input, max_length=1000):
        new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1) if self.chat_history_ids is not None else new_input_ids

        self.chat_history_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )

        output = self.chat_history_ids[:, input_ids.shape[-1]:]
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
