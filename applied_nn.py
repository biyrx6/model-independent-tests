# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import chi2, kstest
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# =============== 可复现 & 设备 ===============
SEED = 718
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# =============== 数据路径与特征列 ===============
INPUT_CSV = r"C:\Users\郭嘉英\OneDrive\Desktop\PHYSICAL_DATA_log.csv"  # ← 改成你的实际路径
feature_cols = [
    "tau_pt","tau_eta","tau_phi",
    "lep_pt","lep_eta","lep_phi",
    "met","met_phi","met_sumet",
    "lead_pt","lead_eta",
    "sublead_pt","sublead_eta","sublead_phi",
    "all_pt"
]
label_col  = "Label"  # 's' / 'b'

# =============== 实验设置 ===============
N_R   = 40403                     # 参考集 R：固定这么多背景
N_R0  = 20000                     # 实验集 W 的 Poisson 均值
LAMBDAS   = [0.1]  # 扫一些 λ
NUM_TOYS  = 50                    # 每个 λ 的 toy Y数
USE_REPLACE_SAMPLING = True       # 有放回抽样（推荐；toys 独立同分布）

# =============== 网络超参（15→64→32→1） ===============
INPUT_DIM = len(feature_cols)   # 15
H1, H2    = 24, 12
OUTPUT_DIM = 1
EPOCHS    = 30000
LR        = 1e-3
CLIP_VALUE = 0.22
PRINT_EVERY_EPOCH = 10000

# 自由度（参数个数）
DF_FIXED = (INPUT_DIM*H1 + H1) + (H1*H2 + H2) + (H2*OUTPUT_DIM + OUTPUT_DIM)
print("DOF =", DF_FIXED)

# =============== 读取数据，分 R/W 池 ===============
df = pd.read_csv(INPUT_CSV)
# 只保留 15 个特征 + Label，并把特征转 float32
df = df[feature_cols + [label_col]].dropna().copy()
df[feature_cols] = df[feature_cols].astype(np.float32)

# 背景池 / 信号池
df_bkg = df[df[label_col] == 'b'].reset_index(drop=True)
df_sig = df[df[label_col] == 's'].reset_index(drop=True)

if len(df_bkg) < N_R:
    raise ValueError(f"背景池不足：需要 N_R={N_R}，但只有 {len(df_bkg)} 条 'b'。")

# 固定参考集 R：从背景池随机取 N_R 个（固定一次）
rng = np.random.default_rng(SEED)
idx_R = rng.choice(len(df_bkg), size=N_R, replace=False)
X_R_np = df_bkg.loc[idx_R, feature_cols].to_numpy(np.float32)  # (N_R, 15)
X_R = torch.from_numpy(X_R_np).to(device)

# 实验池：去掉已经用于 R 的背景，其余背景 + 全部信号
bkg_pool = df_bkg.drop(index=idx_R).reset_index(drop=True)[feature_cols].to_numpy(np.float32)
sig_pool = df_sig[feature_cols].to_numpy(np.float32)

# 参考项权重：w_e = N_R0 / N_R
w_e_scalar = float(N_R0) / float(N_R)

# =============== 网络结构 ===============
class SimpleNN(nn.Module):
    def __init__(self, clip_val):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, OUTPUT_DIM)
        self.act = nn.ReLU()
        self.clip_val = clip_val

    def forward(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        return self.fc3(h2)

    def clip_weights(self):
        for p in self.parameters():
            p.data.clamp_(-self.clip_val, self.clip_val)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============== 采样工具（等概率；不使用 Weight） ===============
def sample_pool(pool: np.ndarray, n: int, replace: bool) -> np.ndarray:
    if n <= 0:
        return np.empty((0, pool.shape[1]), dtype=np.float32)
    if replace:
        idx = rng.integers(0, len(pool), size=n)
    else:
        if n > len(pool):
            raise RuntimeError(f"不放回抽样需要 {n}，但池子只剩 {len(pool)}")
        idx = rng.choice(len(pool), size=n, replace=False)
    return pool[idx]

# =============== 生成单个 toy 的 W（R 已经固定） ===============
def generate_W(lambda_val: float, replace: bool = True):
    # M ~ Poisson(N_R0)
    M = rng.poisson(N_R0)
    if M == 0:
        X_W = np.empty((0, INPUT_DIM), dtype=np.float32)
    else:
        n_sig = rng.binomial(M, lambda_val)
        n_bkg = M - n_sig
        Xs = sample_pool(sig_pool, n_sig, replace)
        Xb = sample_pool(bkg_pool, n_bkg, replace)
        X_W = np.vstack([Xs, Xb]).astype(np.float32)
        # 打乱
        perm = rng.permutation(len(X_W))
        X_W = X_W[perm]
    # 按你的要求：W 内部不区分 s/b，**全标 1**
    y_W = np.ones((X_W.shape[0],), dtype=np.float32)
    return torch.from_numpy(X_W).to(device), torch.from_numpy(y_W).to(device)

# =============== NPLM 损失 ===============
def nplm_unified_loss(fZ, yZ, w_ref_scalar):
    # L = sum[(1-y) * w_e * (exp(f)-1) - y * f]
    return torch.sum((1.0 - yZ) * w_ref_scalar * (torch.exp(fZ) - 1.0) - yZ * fZ)

# =============== 训练一个 toy 并返回 T = -2*min L ===============
def train_and_score_one_toy(lambda_val: float):
    X_W, y_W = generate_W(lambda_val, replace=USE_REPLACE_SAMPLING)
    # 组 Z = R ∪ W，R 全 0，W 全 1
    X_Z = torch.cat([X_R, X_W], dim=0)
    y_Z = torch.cat([
        torch.zeros((X_R.shape[0],), dtype=torch.float32, device=device),  # R -> 0
        y_W                                                                # W -> 1
    ], dim=0)

    model = SimpleNN(CLIP_VALUE).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad(set_to_none=True)
        fZ = model(X_Z).squeeze()
        loss = nplm_unified_loss(fZ, y_Z, w_e_scalar)
        loss.backward()
        opt.step()
        model.clip_weights()
        if PRINT_EVERY_EPOCH and epoch % PRINT_EVERY_EPOCH == 0:
            print(f"[λ={lambda_val:>4.2f}] Epoch {epoch:6d}/{EPOCHS}  loss = {loss.item():.6f}")

    with torch.no_grad():
        fZ = model(X_Z).squeeze()
        final_loss = nplm_unified_loss(fZ, y_Z, w_e_scalar).item()
        T = -2.0 * final_loss
    return T, count_params(model)

# =============== 主流程 ===============
def main():
    print(f"\nN_R = {N_R}, N_R0 = {N_R0}, w_e = {w_e_scalar:.6f}, DOF = {DF_FIXED}")
    results = {}
    for lam in LAMBDAS:
        print(f"\n--- λ = {lam:.2f} ---")
        Ts = []
        dof_once = None
        for i in range(1, NUM_TOYS + 1):
            T, dof = train_and_score_one_toy(lam)
            Ts.append(T)
            dof_once = dof_once or dof
            print(f"  Toy {i:02d}/{NUM_TOYS}: T = {T:.3f}")
        results[lam] = {"Ts": np.asarray(Ts, dtype=float), "dof": dof_once}

    # 只在 λ=0 时做 KS（检验 p 是否近似 U(0,1)）
    if 0.0 in results:
        Ts0 = results[0.0]["Ts"]
        pvals0 = 1.0 - chi2.cdf(Ts0, df=DF_FIXED)
        ks_stat, ks_p = kstest(pvals0, 'uniform')
        print(f"\n[λ=0] KS vs U(0,1): stat = {ks_stat:.4f}, p = {ks_p:.4f}")
    return results
if __name__ == "__main__":
    results = main()

    # ===== 保存 T 值到 JSON（含 DOF 与元信息）=====
    out_dict = {
        "meta": {

            "seed": SEED,
            "device": str(device),
            "epochs": EPOCHS,
            "lr": LR,
            "num_toys": NUM_TOYS,
            "lambdas": LAMBDAS,
            "input_dim": INPUT_DIM,
            "hidden_dim": [H1, H2],
            "output_dim": OUTPUT_DIM,
            "df_fixed": DF_FIXED,
            "input_csv": INPUT_CSV,
            "N_R": N_R,
            "N_R0": N_R0,
            "clip_value": CLIP_VALUE,
            "feature_cols": feature_cols,
            "loss": "sum[(1-y) w_e (exp f - 1) - y f], w_e=N_R0/N_R",
            "sampling": f"Reference fixed; experimental = Poisson({N_R0}), mixture with λ; replace={USE_REPLACE_SAMPLING}"
        },
        "results": {
            str(lam): {
                "Ts": pack["Ts"].tolist(),
                "dof": results[lam]["dof"]
            }
            for lam, pack in results.items()
        }
    }
    with open("T_value_WITHLAM0.1.json", "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2, ensure_ascii=False)
    print("✅ 已保存到 T_value_WITHLAM0.1.json")

    # ====== 可视化：T 直方图 + p-value CDF ======
    for lam, pack in results.items():
        Ts = pack["Ts"]

        # T 直方图 vs 理论 chi^2_DOF
        plt.figure(figsize=(6.4,4.6))
        plt.hist(Ts, bins=25, density=True, alpha=0.45, edgecolor='black', label='Empirical $P(T)$')
        xs = np.linspace(0, max(Ts.max(), 1)*1.1, 600)
        plt.plot(xs, chi2.pdf(xs, df=DF_FIXED), 'k-', lw=2, label=rf'$\chi^2_{{{DF_FIXED}}}$')
        plt.xlabel("T"); plt.ylabel("Density")
        plt.title(f"T histogram (λ={lam})")
        plt.legend(); plt.grid(alpha=0.3, ls='--'); plt.tight_layout(); plt.show()

    # p-value CDF
    plt.figure(figsize=(7,5))
    for lam, pack in results.items():
        Ts = pack["Ts"]
        pvals = 1.0 - chi2.cdf(Ts, df=DF_FIXED)
        s = np.sort(pvals); y = np.arange(1, len(s)+1)/len(s)
        plt.step(s, y, where='post', label=f"λ={lam}")
    plt.plot([0,1],[0,1],'k--',label='Uniform(0,1)')
    plt.xlabel(r"p-value (vs $\chi^2_{%d}$)" % DF_FIXED); plt.ylabel("Empirical CDF")
    plt.title("P-value CDF (15D, R=0 / W=1)")
    plt.legend(loc='lower right'); plt.grid(alpha=0.3, ls='--')
    plt.tight_layout(); plt.show()
