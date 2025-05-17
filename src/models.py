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

