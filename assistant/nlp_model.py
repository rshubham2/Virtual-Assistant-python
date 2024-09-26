from transformers import BartForConditionalGeneration, BartTokenizer
import torch

class NLPModel:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_response(self, input_text):
        inputs = self.tokenizer([input_text], max_length=1024, return_tensors="pt").to(self.device)
        summary_ids = self.model.generate(inputs["input_ids"], num_beams=4, min_length=5, max_length=40, early_stopping=True)
        response = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return response
