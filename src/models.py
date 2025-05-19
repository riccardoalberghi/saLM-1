from model_loader import ModelInfo

MODELS = [
    
    ModelInfo(
        id="qwen-25-15b",
        name="Qwen 2.5 1.5B",
        path="Qwen/Qwen2.5-1.5B",
        parameters=1_500_000_000,
        task="General-purpose text generation"
    ),
    ModelInfo(
        id="quen-25-15b-med",
        name="Qwen 2.5 1.5B Medical Finetuned",
        path="Arthur-77/QWEN2.5-1.5B-medical-finetuned",
        parameters=1_500_000_000,
        task="Medical question answering"
    ),
    ModelInfo(
        id="qwen-25-15b-math",
        name="Qwen 2.5 1.5B Math",
        path="Qwen/Qwen2.5-Math-1.5B",
        parameters=1_500_000_000,
        task="Mathematical reasoning"
    )
]

""" MODELS = [
    ModelInfo(
        id="LLama-3_2-1B-General-lora_model",
        name="Llama-3.2-1B General (LoRA)",
        path="bunnycore/LLama-3.2-1B-General-lora_model",
        parameters=1_000_000_000,
        task="General-purpose text generation (LoRA adapter)"
    ),
    ModelInfo(
        id="llama-3_2-1b-gsm8k-full-finetuning",
        name="Llama-3_2-1B GSM8K Full Finetuning",
        path="axel-datos/Llama-3.2-1B_gsm8k_full-finetuning",
        parameters=1_000_000_000,
        task="Mathematical reasoning (GSM8K)"
    ),
    ModelInfo(
        id="Johhny1201/llama3_2_1b_med_QA_2",
        name="Llama-3_2-1B med QA2",
        path="Johhny1201/llama3.2_1b_med_QA_2",
        parameters=1_000_000_000,
        task="MedQA"
    ),
    
]
 """
