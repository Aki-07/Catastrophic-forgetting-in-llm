
"""
LLM Catastrophic Forgetting — Interactive Demo 
===================================================================
Model: distilgpt2
Data : opus_books en–fr (Hugging Face Datasets)

Usage examples:
  # Fast full run + interactive Q&A
  python llm_forgetting_demo_interactive.py --fast --interactive

  # Full with one epoch each and then Q&A
  python llm_forgetting_demo_interactive.py --epochs_a 1 --epochs_b 1 --interactive
"""

import os, math, random, argparse, json
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm

def device_auto(cpu_flag: bool):
    if cpu_flag:
        return torch.device("cpu")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

#prep the data for our usecase

def get_en_fr(opus_split="train", sample_en=4000, sample_fr=4000, seed=7):
    ds = load_dataset("opus_books", "en-fr", split=opus_split)
    ds = ds.shuffle(seed=seed)
    en_texts, fr_texts = [], []
    for r in ds:
        t = r["translation"]
        if "en" in t and "fr" in t:
            en = (t["en"] or "").strip()
            fr = (t["fr"] or "").strip()
            if len(en) > 20:
                en_texts.append(en)
            if len(fr) > 20:
                fr_texts.append(fr)
        if len(en_texts) >= sample_en and len(fr_texts) >= sample_fr:
            break
    return en_texts[:sample_en], fr_texts[:sample_fr]

def tokenize_texts(tokenizer, texts, block_size=128):
    enc = tokenizer(
        texts,
        return_tensors="pt",      
        truncation=True,
        max_length=block_size,
        padding="max_length"       
    )
    input_ids = enc["input_ids"]
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def make_dataset(tokenizer, texts, block_size=128):
    tokenized = tokenize_texts(tokenizer, texts, block_size)
    class SimpleDS(torch.utils.data.Dataset):
        def __init__(self, data):
            self.input_ids = data["input_ids"]
            self.labels = data["labels"]
        def __len__(self): 
            return len(self.input_ids)
        def __getitem__(self, i):
            return {
                "input_ids": self.input_ids[i],
                "labels": self.labels[i]
            }
    return SimpleDS(tokenized)


@torch.no_grad()
def perplexity(model, tokenizer, texts: List[str], device, block_size=128, max_batches=40):
    model.eval()
    ds = make_dataset(tokenizer, texts, block_size)
    
    # Create a proper data collator that handles padding
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,  # We're doing causal LM, not masked LM
        return_tensors="pt"
    )
    
    dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=8, 
        shuffle=False,
        collate_fn=data_collator  # Use the proper collator
    )
    
    n, total_loss = 0, 0.0
    for i, batch in enumerate(dl):
        if i >= max_batches: 
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        n += 1
    
    if n == 0: 
        return float("inf")
    mean_loss = total_loss / n
    return float(math.exp(mean_loss))

def add_lora(model, r=8, alpha=16, dropout=0.05):
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn","c_proj"]
    )
    model = get_peft_model(model, config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model



def train_lm(model, tokenizer, train_texts, output_dir, device,
             epochs=1, lr=5e-5, batch_size=8, block_size=128, save_total_limit=1):
    os.makedirs(output_dir, exist_ok=True)
    ds = make_dataset(tokenizer, train_texts, block_size)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=20,
        save_steps=200,
        save_total_limit=save_total_limit,
        report_to=[],
        fp16=False, bf16=False
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=data_collator)
    model.to(device)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def generate_once(model, tokenizer, prompt, device, max_new_tokens=80, temperature=0.7):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

def build_models_for_interactive(model_name, device):
    """Load all checkpoints (if available) and return dict {key: model}."""
    models = {}
    try:
        models["baseline"] = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    except Exception as e:
        print("Could not load baseline:", e)
    if os.path.isdir("checkpoints/phaseA_fr"):
        try:
            models["after_A"] = AutoModelForCausalLM.from_pretrained("checkpoints/phaseA_fr").to(device)
        except Exception as e:
            print("Warn: cannot load after_A:", e)
    if os.path.isdir("checkpoints/phaseB_en_no_replay"):
        try:
            models["after_B_no_replay"] = AutoModelForCausalLM.from_pretrained("checkpoints/phaseB_en_no_replay").to(device)
        except Exception as e:
            print("Warn: cannot load after_B_no_replay:", e)
    if os.path.isdir("checkpoints/phaseB_en_replay"):
        try:
            models["after_B_with_replay"] = AutoModelForCausalLM.from_pretrained("checkpoints/phaseB_en_replay").to(device)
        except Exception as e:
            print("Warn: cannot load after_B_with_replay:", e)
    if os.path.isdir("checkpoints/loraA_fr"):
        try:
            models["lora_fr"] = AutoPeftModelForCausalLM.from_pretrained("checkpoints/loraA_fr").to(device)
        except Exception as e:
            print("Warn: cannot load lora_fr:", e)
    if os.path.isdir("checkpoints/loraB_en"):
        try:
            models["lora_en"] = AutoPeftModelForCausalLM.from_pretrained("checkpoints/loraB_en").to(device)
        except Exception as e:
            print("Warn: cannot load lora_en:", e)
    return models

def interactive_cli(tokenizer, models, device):
    keys = list(models.keys())
    if not keys:
        print("No models loaded. Train first or ensure checkpoints/ exist."); return
    print("\nInteractive Q&A — choose a model phase and ask anything.")
    print("Available models:", ", ".join(keys))
    print("Commands: /model <name>  |  /temp <0.1-1.5>  |  /len <tokens>  |  /exit\n")
    current = keys[0]; temp = 0.7; length = 80
    while True:
        cmd = input(f"[{current} | T={temp} | L={length}] > ").strip()
        if not cmd: continue
        if cmd == "/exit": break
        if cmd.startswith("/model "):
            name = cmd.split(" ",1)[1].strip()
            if name in models:
                current = name; print(f"→ Switched to {current}")
            else:
                print("Unknown model. Options:", ", ".join(keys))
            continue
        if cmd.startswith("/temp "):
            try:
                temp = float(cmd.split(" ",1)[1].strip()); print(f"→ temperature = {temp}")
            except: print("Give a number, e.g., /temp 0.8")
            continue
        if cmd.startswith("/len "):
            try:
                length = int(cmd.split(" ",1)[1].strip()); print(f"→ max_new_tokens = {length}")
            except: print("Give an integer, e.g., /len 64")
            continue
        model = models[current]
        out = generate_once(model, tokenizer, cmd, device, max_new_tokens=length, temperature=temp)
        print("\n--- OUTPUT ---")
        print(out)
        print("--------------\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="distilgpt2")
    ap.add_argument("--epochs_a", type=int, default=1)
    ap.add_argument("--epochs_b", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--sample_en", type=int, default=3000)
    ap.add_argument("--sample_fr", type=int, default=3000)
    ap.add_argument("--replay_ratio", type=float, default=0.1)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="Skip all training; only interactive if requested")
    ap.add_argument("--interactive", action="store_true", help="Start interactive Q&A at the end")
    args = ap.parse_args()

    if args.fast:
        args.epochs_a = max(1, args.epochs_a)
        args.epochs_b = max(1, args.epochs_b)
        args.sample_en = min(args.sample_en, 1200)
        args.sample_fr = min(args.sample_fr, 1200)
        print("⚡ Fast mode enabled")

    device = device_auto(args.cpu)
    print(f"device = {device}")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    print("🔤 Loading tokenizer/model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    if not args.skip_train:
        print("Loading EN/FR texts (opus_books)...")
        need_en = args.sample_en + 800
        need_fr = args.sample_fr + 800

        en_all, fr_all = get_en_fr("train", sample_en=need_en, sample_fr=need_fr, seed=7)

        # Safety: if the dataset is smaller than requested, shrink eval size
        eval_en = min(800, max(0, len(en_all) - args.sample_en))
        eval_fr = min(800, max(0, len(fr_all) - args.sample_fr))

        en_train = en_all[:args.sample_en]
        fr_train = fr_all[:args.sample_fr]
        en_eval  = en_all[-eval_en:] if eval_en > 0 else en_all[: min(800, len(en_all))]
        fr_eval  = fr_all[-eval_fr:] if eval_fr > 0 else fr_all[: min(800, len(fr_all))]

        print(f"Train sizes  — EN: {len(en_train)}  FR: {len(fr_train)}")
        print(f"Eval  sizes  — EN: {len(en_eval)}   FR: {len(fr_eval)}")
        # Baseline PPL
        ppl_base_en = perplexity(base_model, tokenizer, en_eval, device, block_size=args.block_size)
        ppl_base_fr = perplexity(base_model, tokenizer, fr_eval, device, block_size=args.block_size)
        print(f"📈 Baseline PPL — EN: {ppl_base_en:.2f} | FR: {ppl_base_fr:.2f}")

        # Phase A
        print("\n Phase A — Fine-tune on FR")
        train_lm(base_model, tokenizer, fr_train, "checkpoints/phaseA_fr", device,
                 epochs=args.epochs_a, lr=args.lr, batch_size=args.batch_size, block_size=args.block_size)
        model_A = AutoModelForCausalLM.from_pretrained("checkpoints/phaseA_fr").to(device)

        ppl_A_en = perplexity(model_A, tokenizer, en_eval, device, block_size=args.block_size)
        ppl_A_fr = perplexity(model_A, tokenizer, fr_eval, device, block_size=args.block_size)
        print(f"📈 After A — EN: {ppl_A_en:.2f} | FR: {ppl_A_fr:.2f}")

        # Phase B (no replay)
        print("\n Phase B — Continue on EN (no replay)")
        train_lm(model_A, tokenizer, en_train, "checkpoints/phaseB_en_no_replay", device,
                 epochs=args.epochs_b, lr=args.lr, batch_size=args.batch_size, block_size=args.block_size)
        model_B = AutoModelForCausalLM.from_pretrained("checkpoints/phaseB_en_no_replay").to(device)
        ppl_B_en = perplexity(model_B, tokenizer, en_eval, device, block_size=args.block_size)
        ppl_B_fr = perplexity(model_B, tokenizer, fr_eval, device, block_size=args.block_size)
        print(f"📉 After B (no replay) — EN: {ppl_B_en:.2f} | FR: {ppl_B_fr:.2f}  ← forgetting expected")

        # Phase B (replay)
        print("\n Phase B (Replay) — mix {0:.0%} FR into EN".format(args.replay_ratio))
        mix_n_fr = int(len(en_train) * args.replay_ratio)
        mix_fr = fr_train[:mix_n_fr]
        replay_mixed = en_train + mix_fr
        random.shuffle(replay_mixed)

        model_A_re = AutoModelForCausalLM.from_pretrained("checkpoints/phaseA_fr").to(device)
        train_lm(model_A_re, tokenizer, replay_mixed, "checkpoints/phaseB_en_replay", device,
                 epochs=args.epochs_b, lr=args.lr, batch_size=args.batch_size, block_size=args.block_size)
        model_Br = AutoModelForCausalLM.from_pretrained("checkpoints/phaseB_en_replay").to(device)
        ppl_Br_en = perplexity(model_Br, tokenizer, en_eval, device, block_size=args.block_size)
        ppl_Br_fr = perplexity(model_Br, tokenizer, fr_eval, device, block_size=args.block_size)
        print(f"✅ After B (replay) — EN: {ppl_Br_en:.2f} | FR: {ppl_Br_fr:.2f}  ← less forgetting")

        # LoRA isolation
        print("\n🧩 Isolation via LoRA adapters — separate FR/EN adapters")
        base_for_lora = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        base_for_lora = add_lora(base_for_lora)
        train_lm(base_for_lora, tokenizer, fr_train, "checkpoints/loraA_fr", device,
                 epochs=args.epochs_a, lr=args.lr, batch_size=args.batch_size, block_size=args.block_size)

        base_for_lora_B = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        base_for_lora_B = add_lora(base_for_lora_B)
        train_lm(base_for_lora_B, tokenizer, en_train, "checkpoints/loraB_en", device,
                 epochs=args.epochs_b, lr=args.lr, batch_size=args.batch_size, block_size=args.block_size)

    # Interactive CLI
    if args.interactive:
        models = build_models_for_interactive(args.model, device)
        interactive_cli(tokenizer, models, device)
    else:
        print("\nTip: run with --interactive to start an interactive Q&A CLI.")

if __name__ == "__main__":
    main()
