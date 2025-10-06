# training/train_lora.py  (fast demo edition)
import os, json, time, math, random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prompt_utils import to_chat, simple_template, DEFAULT_SYSTEM

BASE_DIR = Path("/mnt/project/atharva-dental-assistant")
DATA_DIR = BASE_DIR / "datasets" / "training"

# ------------------------------
# Demo-friendly defaults (override via env)
# ------------------------------
BASE_MODEL   = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
MAX_SEQ_LEN  = int(os.environ.get("MAX_SEQ_LEN", "256"))     # ↓ from 512
LORA_R       = int(os.environ.get("LORA_R", "4"))            # ↓ from 8
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", "8"))        # ↓ from 16
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
LR           = float(os.environ.get("LR", "2e-4"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.02"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "4"))        # ↓ from 8
MAX_STEPS    = int(os.environ.get("MAX_STEPS", "80"))        # ↓ from 400 (≈5–10 min)
# Optional dataset subsample for speed (0 = use all)
DEMO_MAX_TRAIN_SAMPLES = int(os.environ.get("DEMO_MAX_TRAIN_SAMPLES", "0"))
DEMO_MAX_VAL_SAMPLES   = int(os.environ.get("DEMO_MAX_VAL_SAMPLES", "0"))

OUTPUT_ROOT = BASE_DIR / "artifacts" / "train" / time.strftime("%Y%m%d-%H%M%S")

# Use all CPU cores for a faster demo
torch.set_num_threads(max(1, os.cpu_count()))

print(f"Base model: {BASE_MODEL}")
print(f"Output dir: {OUTPUT_ROOT}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map=None
)
model = prepare_model_for_kbit_training(model)  # safe on CPU

# Trim LoRA to attention projections only (fewer trainable params)
peft_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)

def build_example(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    msgs = to_chat(messages, DEFAULT_SYSTEM)
    use_chat_template = hasattr(tokenizer, "apply_chat_template")
    text_prompt = (
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        if use_chat_template else simple_template(msgs)
    )
    # Find assistant content for label masking
    assistant_text = [m["content"] for m in msgs if m["role"]=="assistant"][-1]
    _ = assistant_text.strip()

    tok = tokenizer(text_prompt, truncation=True, max_length=MAX_SEQ_LEN, padding=False, return_tensors=None)

    prefix_msgs = [m for m in msgs if m["role"]!="assistant"]
    prefix_text = (
        tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True)
        if use_chat_template else simple_template(prefix_msgs) + "\n[ASSISTANT]\n"
    )
    prefix_tok = tokenizer(prefix_text, truncation=True, max_length=MAX_SEQ_LEN, padding=False, return_tensors=None)

    input_ids = tok["input_ids"]
    labels = input_ids.copy()
    mask_len = min(len(prefix_tok["input_ids"]), len(labels))
    labels[:mask_len] = [-100] * mask_len
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1]*len(input_ids)}

def load_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        yield json.loads(line)

# ------------------------------
# Load + optional subsample
# ------------------------------
train_records = list(load_jsonl(DATA_DIR/"train.jsonl"))
val_records   = list(load_jsonl(DATA_DIR/"val.jsonl"))

if DEMO_MAX_TRAIN_SAMPLES > 0 and len(train_records) > DEMO_MAX_TRAIN_SAMPLES:
    random.seed(42)
    train_records = random.sample(train_records, DEMO_MAX_TRAIN_SAMPLES)
if DEMO_MAX_VAL_SAMPLES > 0 and len(val_records) > DEMO_MAX_VAL_SAMPLES:
    random.seed(123)
    val_records = random.sample(val_records, DEMO_MAX_VAL_SAMPLES)

train_ds = [build_example(rec["messages"]) for rec in train_records]
val_ds   = [build_example(rec["messages"]) for rec in val_records]

@dataclass
class Collator:
    pad_token_id: int = tokenizer.pad_token_id
    def __call__(self, batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attn = [], [], []
        for x in batch:
            pad    = [self.pad_token_id] * (maxlen - len(x["input_ids"]))
            maskpd = [0] * (maxlen - len(x["attention_mask"]))
            lblpd  = [-100] * (maxlen - len(x["labels"]))  # ← fixed missing ')'
            input_ids.append(x["input_ids"] + pad)
            labels.append(x["labels"] + lblpd)
            attn.append(x["attention_mask"] + maskpd)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Training args: no eval during train, single final save, fewer steps
# ------------------------------
args = TrainingArguments(
    output_dir=str(OUTPUT_ROOT),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    max_steps=MAX_STEPS,
    lr_scheduler_type="cosine",
    logging_steps=50,
    evaluation_strategy="no",         # ← no eval to save time
    save_steps=10_000_000,            # ← avoid mid-run checkpoints
    save_total_limit=1,               # ← keep only final
    bf16=False, fp16=False,
    dataloader_num_workers=0,
    report_to="none"
)

# Helpful run summary
N = len(train_ds)
steps_per_epoch = max(1, math.ceil(N / (BATCH_SIZE * GRAD_ACCUM)))
est_epochs = args.max_steps / steps_per_epoch
print(f"Train examples: {N}, steps/epoch: {steps_per_epoch}, "
      f"optimizer steps: {args.max_steps}, ~epochs: {est_epochs:.2f}")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=None,    # no eval during training
    data_collator=Collator(),
)

trainer.train()

# Save adapter + tokenizer
model.save_pretrained(str(OUTPUT_ROOT / "lora_adapter"))
tokenizer.save_pretrained(str(OUTPUT_ROOT / "tokenizer"))

# Run manifest
(OUTPUT_ROOT / "run.json").write_text(json.dumps({
    "base_model": BASE_MODEL,
    "max_seq_len": MAX_SEQ_LEN,
    "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
    "lr": LR, "warmup_ratio": WARMUP_RATIO,
    "batch": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "max_steps": MAX_STEPS,
    "demo_max_train_samples": DEMO_MAX_TRAIN_SAMPLES,
    "demo_max_val_samples": DEMO_MAX_VAL_SAMPLES
}, indent=2), encoding="utf-8")

print(f"Training complete. Artifacts at {OUTPUT_ROOT}")
