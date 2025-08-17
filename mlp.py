import argparse, random, os, json
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


def device_auto(cpu_flag: bool):
    if cpu_flag:
        return torch.device("cpu")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class TaskData:
    X_train: torch.Tensor; y_train: torch.Tensor
    X_val: torch.Tensor; y_val: torch.Tensor

def make_task(dataset: str, n_samples=4000, noise=0.2, test_size=0.25, seed=0) -> TaskData:
    if dataset == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif dataset == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    else:
        raise ValueError("Unknown dataset")
    Xa, Xb, ya, yb = train_test_split(X.astype(np.float32), y.astype(np.int64),
                                      test_size=test_size, random_state=seed)
    return TaskData(torch.from_numpy(Xa), torch.from_numpy(ya),
                    torch.from_numpy(Xb), torch.from_numpy(yb))

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(dropout)
    def features(self, x):
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); return x
    def forward(self, x):
        return self.fc3(self.features(x))

class AdapterHead(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.down = nn.Linear(hidden, 16)
        self.up = nn.Linear(16, hidden)
    def forward(self, h):
        return self.up(F.relu(self.down(h)))

def accuracy(model, dl, device):
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb); preds = logits.argmax(1)
            correct += (preds==yb).sum().item(); total += yb.numel()
    return correct / max(1,total)

def make_loader(task: TaskData, bs=128, shuffle=True):
    return DataLoader(TensorDataset(task.X_train, task.y_train), batch_size=bs, shuffle=shuffle)
def make_valloader(task: TaskData, bs=256):
    return DataLoader(TensorDataset(task.X_val, task.y_val), batch_size=bs, shuffle=False)

def train_simple(model, dl, val_dl, device, epochs=20, lr=1e-3, desc="Training"):
    model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    history=[]
    for _ in tqdm(range(epochs), desc=desc, unit="ep"):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        history.append(accuracy(model, val_dl, device))
    return history

def train_with_replay(model, dl_b, val_a, val_b, device, buffer, epochs=20, lr=1e-3, alpha=0.25):
    model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    buf_X, buf_y = buffer; idx = torch.randperm(buf_X.size(0)); ptr=0
    history=[]
    for _ in tqdm(range(epochs), desc="🔁 Training B with Replay", unit="ep"):
        model.train()
        for xb, yb in dl_b:
            xb, yb = xb.to(device), yb.to(device)
            k = int(alpha*xb.size(0))
            if k>0:
                if ptr+k >= idx.numel(): idx = torch.randperm(buf_X.size(0)); ptr=0
                sel = idx[ptr:ptr+k]; ptr += k
                x_re, y_re = buf_X[sel].to(device), buf_y[sel].to(device)
                xb = torch.cat([xb[:xb.size(0)-k], x_re], 0)
                yb = torch.cat([yb[:yb.size(0)-k], y_re], 0)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        history.append((accuracy(model, val_a, device), accuracy(model, val_b, device)))
    return history

def estimate_fisher(model, dl, device, samples=1024):
    model.eval()
    fisher = {n: torch.zeros_like(p, device=device) for n,p in model.named_parameters() if p.requires_grad}
    count=0
    with tqdm(total=samples, desc="📏 Estimating Fisher (Task A)", unit="ex") as bar:
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            for i in range(xb.size(0)):
                if count>=samples: break
                model.zero_grad()
                out = model(xb[i:i+1])
                lp = F.log_softmax(out,1)[0, yb[i].item()]
                lp.backward(retain_graph=True)
                for (n,p) in model.named_parameters():
                    if p.grad is not None and p.requires_grad:
                        fisher[n] += p.grad.detach()**2
                count += 1; bar.update(1)
            if count>=samples: break
    for n in fisher: fisher[n] /= max(1,count)
    return fisher

def train_with_ewc(model, dl_b, val_a, val_b, device, fisher, params_star, epochs=20, lr=1e-3, lam=50.0):
    model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    history=[]
    for _ in tqdm(range(epochs), desc="🧷 Training B with EWC", unit="ep"):
        model.train()
        for xb, yb in dl_b:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            reg = 0.0
            for (n,p) in model.named_parameters():
                if n in fisher: reg = reg + (fisher[n] * (p - params_star[n])**2).sum()
            (loss + lam*reg).backward(); opt.step(); opt.zero_grad()
        history.append((accuracy(model, val_a, device), accuracy(model, val_b, device)))
    return history

def train_with_adapters(base, adapters, head, dl, val_dl, device, epochs=20, lr=1e-3):
    base.to(device); adapters.to(device); head.to(device)
    for p in base.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(list(adapters.parameters()) + list(head.parameters()), lr=lr)
    history=[]
    for _ in tqdm(range(epochs), desc="🧩 Training Adapters", unit="ep"):
        base.train(); adapters.train(); head.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            h = base.features(xb); h = h + adapters(h)
            loss = F.cross_entropy(head(h), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        base.eval(); adapters.eval(); head.eval()
        with torch.no_grad():
            acc = 0.0; tot=0
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                h = base.features(xb); h = h + adapters(h)
                preds = head(h).argmax(1)
                acc += (preds==yb).sum().item(); tot += yb.numel()
        history.append(acc/max(1,tot))
    return history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_a", type=int, default=25)
    ap.add_argument("--epochs_b", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--replay-buf", dest="replay_buf", type=int, default=1000)
    ap.add_argument("--replay-alpha", dest="replay_alpha", type=float, default=0.25)
    ap.add_argument("--ewc-lambda", dest="ewc_lambda", type=float, default=50.0)
    ap.add_argument("--fast", action="store_true", help="Quick demo: 12 epochs each, smaller buffer")
    args = ap.parse_args()

    if args.fast:
        args.epochs_a = max(args.epochs_a, 12)
        args.epochs_b = max(args.epochs_b, 12)
        args.replay_buf = min(args.replay_buf, 600)
        print("⚡ Fast mode: epochs_a=12, epochs_b=12, replay_buf<=600")

    device = device_auto(args.cpu)
    print(f"\n Using device: {device}")
    os.makedirs("outputs", exist_ok=True)
    set_seed(args.seed)

    print("\n📦 Creating tasks (A=moons, B=circles)...")
    A = make_task("moons", seed=args.seed)
    B = make_task("circles", seed=args.seed+1)
    dlA, dlB = make_loader(A), make_loader(B)
    vA, vB = make_valloader(A), make_valloader(B)

    # Baseline 
    print("\n🧪 Phase 1 — Train Task A (baseline)")
    model = MLP(hidden=64, dropout=0.1)
    histA = train_simple(model, dlA, vA, device, args.epochs_a, args.lr, desc="🅰️  Training Task A")
    print(f"Task A accuracy after A: {histA[-1]:.3f}")

    print("\n🧪 Phase 2 — Train Task B (no protection)")
    histB = train_simple(model, dlB, vB, device, args.epochs_b, args.lr, desc="🅱️  Training Task B")
    accA_afterB = accuracy(model, vA, device); accB_after = accuracy(model, vB, device)
    print(f"Task A AFTER B (forgetting): {accA_afterB:.3f} | Task B: {accB_after:.3f}")

    # Replay
    print("\n🔁 Fix 1 — Replay (tiny rehearsal buffer)")
    model_r = MLP(hidden=64, dropout=0.1)
    _ = train_simple(model_r, dlA, vA, device, args.epochs_a, args.lr, desc="🅰️  Pretrain on A")
    buf_idx = torch.randperm(A.X_train.size(0))[:args.replay_buf]
    buffer = (A.X_train[buf_idx], A.y_train[buf_idx])
    hist_replay = train_with_replay(model_r, dlB, vA, vB, device, buffer,
                                    epochs=args.epochs_b, lr=args.lr, alpha=args.replay_alpha)
    accA_replay, accB_replay = hist_replay[-1]
    print(f"Replay retained A: {accA_replay:.3f} | B: {accB_replay:.3f}  (buf={args.replay_buf}, alpha={args.replay_alpha})")

    # EWC
    print("\n🧷 Fix 2 — Elastic Weight Consolidation (EWC)")
    model_e = MLP(hidden=64, dropout=0.1)
    _ = train_simple(model_e, dlA, vA, device, args.epochs_a, args.lr, desc="🅰️  Pretrain on A")
    print("   📏 Estimating Fisher from Task A...")
    params_star = {n: p.detach().clone() for n,p in model_e.named_parameters() if p.requires_grad}
    fisher = estimate_fisher(model_e, dlA, device, samples=1024)
    hist_ewc = train_with_ewc(model_e, dlB, vA, vB, device, fisher, params_star,
                              epochs=args.epochs_b, lr=args.lr, lam=args.ewc_lambda)
    accA_ewc, accB_ewc = hist_ewc[-1]
    print(f"EWC retained A: {accA_ewc:.3f} | B: {accB_ewc:.3f}  (lambda={args.ewc_lambda})")

    # Isolation via adapters
    print("\n🧩 Fix 3 — Parameter Isolation (task-specific adapters)")
    base = MLP(hidden=64, dropout=0.1)
    headA, adaptersA = nn.Linear(64,2), AdapterHead(64)
    _ = train_with_adapters(base, adaptersA, headA, dlA, vA, device, epochs=args.epochs_a, lr=args.lr)
    headB, adaptersB = nn.Linear(64,2), AdapterHead(64)
    _ = train_with_adapters(base, adaptersB, headB, dlB, vB, device, epochs=args.epochs_b, lr=args.lr)

    def eval_with(adapters, head, vdl):
        base.eval(); adapters.eval(); head.eval()
        c=t=0
        with torch.no_grad():
            for xb, yb in vdl:
                xb, yb = xb.to(device), yb.to(device)
                h = base.features(xb); h = h + adapters(h)
                c += (head(h).argmax(1)==yb).sum().item(); t += yb.numel()
        return c/max(1,t)

    accA_iso_after = eval_with(adaptersA, headA, vA)
    accB_iso_after = eval_with(adaptersB, headB, vB)
    print(f"Isolation (adapters) — A: {accA_iso_after:.3f} | B: {accB_iso_after:.3f}")

    print("\n Saving plots and results to ./outputs ...")
    plt.figure(); plt.plot(histA); plt.title("Task A (val acc)"); plt.xlabel("epochs"); plt.ylabel("acc"); plt.savefig("outputs/01_A.png", dpi=160); plt.close()
    plt.figure(); plt.plot(histB); plt.title("Task B (val acc)"); plt.xlabel("epochs"); plt.ylabel("acc"); plt.savefig("outputs/02_B.png", dpi=160); plt.close()
    plt.figure(); plt.plot([a for a,b in hist_replay], label="A during B (Replay)"); plt.plot([b for a,b in hist_replay], label="B (Replay)"); plt.legend(); plt.title("Replay"); plt.xlabel("epochs"); plt.ylabel("acc"); plt.savefig("outputs/03_replay.png", dpi=160); plt.close()
    plt.figure(); plt.plot([a for a,b in hist_ewc], label="A during B (EWC)"); plt.plot([b for a,b in hist_ewc], label="B (EWC)"); plt.legend(); plt.title("EWC"); plt.xlabel("epochs"); plt.ylabel("acc"); plt.savefig("outputs/04_ewc.png", dpi=160); plt.close()

    import numpy as np
    plt.figure()
    labels = ["Baseline A after B","Replay","EWC","Isolation (A)"]
    vals = [accA_afterB, accA_replay, accA_ewc, accA_iso_after]
    plt.bar(np.arange(len(vals)), vals); plt.xticks(np.arange(len(vals)), labels, rotation=20, ha='right')
    plt.ylim(0,1.0); plt.ylabel("Task A accuracy retained"); plt.title("Retention after Task B")
    plt.tight_layout(); plt.savefig("outputs/05_summary.png", dpi=160); plt.close()

    results = {
        "device": str(device),
        "baseline": {"A_after_B": float(accA_afterB), "B_after_B": float(accB_after)},
        "replay": {"A_after_B": float(accA_replay), "B_after_B": float(accB_replay)},
        "ewc": {"A_after_B": float(accA_ewc), "B_after_B": float(accB_ewc)},
        "isolation": {"A_after_B": float(accA_iso_after), "B_after_B": float(accB_iso_after)},
        "notes": "Fast synthetic demo; adapters isolate parameters by task."
    }
    with open("outputs/results.json","w") as f: json.dump(results, f, indent=2)

    print("\n Done. Key retention (Task A after B):")
    print(f"   Baseline: {results['baseline']['A_after_B']:.3f}")
    print(f"   Replay  : {results['replay']['A_after_B']:.3f}")
    print(f"   EWC     : {results['ewc']['A_after_B']:.3f}")
    print(f"   Isolate : {results['isolation']['A_after_B']:.3f}")

if __name__ == "__main__":
    main()
