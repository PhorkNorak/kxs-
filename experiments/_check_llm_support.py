"""Quick smoke check: does Unsloth load the official Gemma 4 / Qwen 3.5 repos?

Runs in ~1 minute (downloads tokenizer + config, no weights to disk).
"""
from unsloth import FastLanguageModel

CANDIDATES = [
    "google/gemma-4-E4B",
    "Qwen/Qwen3.5-4B",
]

for name in CANDIDATES:
    print(f"\n=== Trying {name} ===")
    try:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=name,
            load_in_4bit=True,
            max_seq_length=128,
        )
        print(f"  UNSLOTH OK: {name}")
        del model, tok
    except Exception as e:
        msg = str(e)[:200]
        print(f"  UNSLOTH FAILED: {type(e).__name__}: {msg}")
        # Try plain transformers fallback
        try:
            from transformers import AutoConfig, AutoTokenizer
            cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            print(f"  HF FALLBACK OK: arch={cfg.architectures} vocab={tok.vocab_size}")
        except Exception as e2:
            print(f"  HF FALLBACK FAILED: {type(e2).__name__}: {str(e2)[:150]}")
