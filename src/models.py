from src.model_loader import ModelInfo

MODELS = [
    ModelInfo(
        id="LLama-3_2-1B-General-lora_model",
        name="Llama-3.2-1B General (LoRA)",
        path="bunnycore/LLama-3.2-1B-General-lora_model",
        parameters=1_000_000_000,
        task="General-purpose text generation (LoRA adapter)"
    ),
    ModelInfo(
        id="llama-3_2-1b-gsm8k-full-finetuning",
        name="Llama-3.2-1B GSM8K Full Finetuning",
        path="axel-datos/Llama-3.2-1B_gsm8k_full-finetuning",
        parameters=1_000_000_000,
        task="Mathematical reasoning (GSM8K)"
    ),
    ModelInfo(
        id="Johhny1201/llama3.2_1b_med_QA_2",
        name="Llama-3.2-1B med QA2",
        path="Johhny1201/llama3.2_1b_med_QA_2",
        parameters=1_000_000_000,
        task="MedQA"
    ),
    
]
