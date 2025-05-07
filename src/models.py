from model_loader import ModelInfo

MODELS = [
    ModelInfo(
        id="gpt-neo-general",
        name="GPT‑Neo‑125M (General)",
        path="EleutherAI/gpt-neo-125m",
        parameters=125_000_000,
        task="General‑purpose text generation"
    ),
    ModelInfo(
        id="gpt-neo-code-clippy",
        name="GPT‑Neo‑125M‑Code‑Clippy",
        path="flax-community/gpt-neo-125M-code-clippy",
        parameters=125_000_000,
        task="Code completion"
    ),
    ModelInfo(
        id="gpt-neo-apps",
        name="GPT‑Neo‑125M‑APPS",
        path="flax-community/gpt-neo-125M-apps",
        parameters=125_000_000,
        task="Programming problem solving"
    ),
    ModelInfo(
        id="gpt-neo-math",
        name="GPT‑Neo‑Math‑Small",
        path="Saisam/gpt-neo-math-small",
        parameters=125_000_000,
        task="Proof‑step generation (Lean tactics)"
    ),
    ModelInfo(
        id="gpt-neo-cs",
        name="GPT‑Neo‑CS‑Finetuned",
        path="KimByeongSu/gpt-neo-125m-cs-finetuning-20000",
        parameters=125_000_000,
        task="Computer science (code & tutorials)"
    ),
    ModelInfo(
        id="gpt-neo-literature",
        name="GPT‑Neo‑Literature",
        path="hakurei/lit-125M",
        parameters=125_000_000,
        task="Fictional storytelling"
    ),
    ModelInfo(
        id="gpt-neo-history",
        name="GPT‑Neo‑History",
        path="shubhamgantayat/EleutherAI-gpt-neo-125m-brief-history-of-time-model",
        parameters=125_000_000,
        task="Historical analysis"
    )
]