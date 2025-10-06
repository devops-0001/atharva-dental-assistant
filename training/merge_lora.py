# training/merge_lora.py
import os, json, shutil, tarfile
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = Path("/mnt/project/atharva-dental-assistant")
ART_ROOT = BASE_DIR / "artifacts" / "train"

BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
RUN_ID = os.environ["RUN_ID"]  # e.g., 20250928-121314 (folder under artifacts/train)

run_dir = ART_ROOT / RUN_ID
adapter_dir = run_dir / "lora_adapter"
tok_dir = run_dir / "tokenizer"
out_dir = run_dir / "merged-model"

assert adapter_dir.exists(), f"Missing adapter at {adapter_dir}"

print(f"Loading base {BASE_MODEL} and merging adapter from {adapter_dir}")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, device_map=None)
merged = PeftModel.from_pretrained(base, str(adapter_dir))
merged = merged.merge_and_unload()  # apply LoRA into base weights
tok = AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else BASE_MODEL, use_fast=True)

out_dir.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(out_dir)
tok.save_pretrained(out_dir)

# Create a tarball for OCI packaging in Lab 3
#tgz_path = run_dir / "model.tgz"
#with tarfile.open(tgz_path, "w:gz") as tar:
#    tar.add(out_dir, arcname="model")
#print(f"Merged model saved at {out_dir}, tarball at {tgz_path}")
