# -*- coding: utf-8 -*-
"""
Profiled NN density-ratio test with nuisance (T = tau - Delta),
with extensive diagnostics printing so you can spot problems without extra checks.
"""
import os, json, random, math, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning)

# =============== Reproducibility & device ===============
SEED = 718
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# =============== Data & columns ===============
INPUT_CSV = r"C:\Users\郭嘉英\OneDrive\Desktop\PHYSICAL_DATA_log.csv"  # ← set your path
feature_cols = [
    "tau_pt","tau_eta","tau_phi",
    "lep_pt","lep_eta","lep_phi",
    "met","met_phi","met_sumet",
    "lead_pt","lead_eta",
    "sublead_pt","sublead_eta","sublead_phi",
    "all_pt"
]
BIN_VAR = "all_pt"   # variable used for binning
label_col  = "Label" # 's' / 'b'

# =============== Experiment setup ===============
N_R   = 40403                 # reference size (fixed)
N_R0  = 20000                 # Poisson mean for the experimental sample
LAMBDAS   = [0.01]            # modify as needed
NUM_TOYS  = 50
USE_REPLACE_SAMPLING = True
WCLIP = 0.22                  # weight clipping bound

# NN
INPUT_DIM = len(feature_cols)  # 15
H1, H2    = 24, 12
OUTPUT_DIM = 1
EPOCHS    = 30000
LR        = 1e-3
PRINT_EVERY_EPOCH = 30000      # print progress every N epochs
LOG_LOSS_COMPONENTS_EVERY = 2000

# For chi2 p-value under H0 we use NN parameter count as df (common heuristic)
DF_FIXED = (INPUT_DIM*H1 + H1) + (H1*H2 + H2) + (H2*OUTPUT_DIM + OUTPUT_DIM)
print("Assumed DOF for chi-square =", DF_FIXED)

# =============== Read & split pools ===============
df = pd.read_csv(INPUT_CSV)
df = df[feature_cols + [label_col]].dropna().copy()
df[feature_cols] = df[feature_cols].astype(np.float32)
df_bkg = df[df[label_col]=='b'].reset_index(drop=True)
df_sig = df[df[label_col]=='s'].reset_index(drop=True)

if len(df_bkg) < N_R:
    raise ValueError(f"Background pool too small: need N_R={N_R}, but only have {len(df_bkg)} rows with label 'b'.")

rng = np.random.default_rng(SEED)
idx_R = rng.choice(len(df_bkg), size=N_R, replace=False)
X_R_np = df_bkg.loc[idx_R, feature_cols].to_numpy(np.float32)
X_R    = torch.from_numpy(X_R_np).to(device)

bkg_pool = df_bkg.drop(index=idx_R).reset_index(drop=True)[feature_cols].to_numpy(np.float32)
sig_pool = df_sig[feature_cols].to_numpy(np.float32)

print(f"Pool sizes: R={len(X_R_np)}, bkg_pool={len(bkg_pool)}, sig_pool={len(sig_pool)}")

# reference weight factor (extended likelihood)
w_e_scalar = float(N_R0) / float(N_R)
print(f"Reference weight w_e = N_R0/N_R = {w_e_scalar:.6f}")

# =============== Binning & bin-fit helpers ===============
def build_bins_from_array(x: np.ndarray, nbins=30, margin=0.05):
    """Build evenly spaced bin edges with a small margin around data range."""
    lo = np.min(x); hi = np.max(x)
    span = hi - lo
    lo -= margin * span; hi += margin * span
    edges = np.linspace(lo, hi, nbins+1, dtype=np.float32)
    return torch.from_numpy(edges).to(device)

def build_binfit_from_R(x_R_1d: np.ndarray, edges: torch.Tensor,
                        n_nuis=2, a0_scale=0.12, a1_std=0.01):
    """
    Use the R histogram to create a positive baseline a0 per bin,
    plus a small linear sensitivity matrix A1 for nuisances.
    """
    nb = edges.numel()-1
    hist, _ = np.histogram(x_R_1d, bins=edges.detach().cpu().numpy())
    freq = hist / max(1, hist.sum())
    a0 = (a0_scale * (freq + 1e-3)).astype(np.float32)  # keep strictly > 0
    A1 = np.random.normal(0.0, a1_std, size=(n_nuis, nb)).astype(np.float32)
    A0_t = torch.from_numpy(a0).to(device)
    A1_t = torch.from_numpy(A1).to(device)
    return A0_t, A1_t, hist

# =============== Layers & model ===============
class BinStepLayer(nn.Module):
    """Hard one-hot binning (frozen, non-trainable)."""
    def __init__(self, edges: torch.Tensor):
        super().__init__()
        edges = edges.detach().clone()
        self.register_buffer("edges", edges)
        self.nbins = edges.numel() - 1
        self.l1 = nn.Linear(1, self.nbins*2, bias=True)
        self.l2 = nn.Linear(self.nbins*2, self.nbins, bias=False)
        W = 100.0
        with torch.no_grad():
            self.l1.weight.zero_(); self.l1.bias.zero_(); self.l2.weight.zero_()
            for i in range(self.nbins+1):
                if i==0:
                    self.l1.weight[2*i,0]=W; self.l1.bias[2*i]=-W*edges[i]
                    self.l2.weight[i,2*i]=1.0
                elif i==self.nbins:
                    self.l1.weight[2*i-1,0]=W; self.l1.bias[2*i-1]=-W*edges[i]
                    self.l2.weight[i-1,2*i-1]=-1.0
                else:
                    self.l1.weight[2*i-1,0]=W; self.l1.bias[2*i-1]=-W*edges[i]
                    self.l1.weight[2*i,0]=W;   self.l1.bias[2*i]=-W*edges[i]
                    self.l2.weight[i,2*i-1]=1.0; self.l2.weight[i-1,2*i]=-1.0
        for p in self.parameters(): p.requires_grad=False

    def forward(self, x1d: torch.Tensor):
        z = self.l1(x1d.view(-1,1))
        z = torch.relu(torch.sign(z))
        z = self.l2(z)
        with torch.no_grad():
            z.copy_((z>0).to(z.dtype))
        return z  # [N, nbins]

class ExpLayerLinear(nn.Module):
    """
    Linear bin-yield model e(ν) = a0 + A1^T ν for each bin, with epsilon clamp to avoid negatives.
    Returns a vector of expected yields per bin given ν.
    """
    def __init__(self, A0: torch.Tensor, A1: torch.Tensor, eps:float=1e-6):
        super().__init__()
        self.register_buffer("a0", A0)      # [nbins]
        self.register_buffer("A1", A1)      # [n_nuis, nbins]
        self.eps = eps

    def forward(self, nu: torch.Tensor):
        e = self.a0 + self.A1.transpose(0,1) @ nu  # [nbins]
        return torch.clamp(e, min=self.eps)

class SimpleNN(nn.Module):
    """Small MLP for f(x); weights are clipped elementwise to ±WCLIP after each step."""
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
        return self.fc3(h2).squeeze(-1)  # f(x)
    def clip_weights(self):
        for p in self.parameters():
            p.data.clamp_(-self.clip_val, self.clip_val)

class NPLM_WithNuis(nn.Module):
    """Full model: hard-binning + linear nuisance yields + NN f(x)."""
    def __init__(self, edges, A0, A1, nu_init, nu_ref, nu0, sigma, clip_val):
        super().__init__()
        self.oi = BinStepLayer(edges)
        self.ei = ExpLayerLinear(A0, A1)
        self.eiR= ExpLayerLinear(A0, A1)
        self.nu = nn.Parameter(nu_init.clone())
        self.register_buffer("nuR", nu_ref.clone())
        self.register_buffer("nu0", nu0.clone())
        self.register_buffer("sig", sigma.clone())
        self.f  = SimpleNN(clip_val)
    def forward(self, X_all, x_bin_1d):
        oi  = self.oi(x_bin_1d)         # [N, nbins]
        fX  = self.f(X_all)             # [N]
        ei  = self.ei(self.nu)          # [nbins]
        eiR = self.eiR(self.nuR)        # [nbins]
        return oi, ei, eiR, fX, self.nu, self.nuR, self.nu0, self.sig

# =============== Loss & components ===============
def npl_loss_with_nuis(y, w_e, outs):
    """
    L = Σ[(1-y) w_e (exp(f + Lbinned) - 1) - y (f + Lbinned)]  - 0.5*((ν-ν0)^2 - (νR-ν0)^2)/σ^2
    """
    oi, ei, eiR, fX, nu, nuR, nu0, sig = outs
    log_ratio = torch.log(ei/eiR)                    # [nbins]
    Lbinned   = torch.sum(oi * log_ratio, dim=1)     # [N]
    Laux = -0.5*torch.sum(((nu-nu0)**2 - (nuR-nu0)**2)/(sig**2))
    return torch.sum((1.0-y)*w_e*(torch.exp(fX+Lbinned)-1.0) - y*(fX+Lbinned)) - Laux

@torch.no_grad()
def decompose_loss_terms(y, w_e, outs):
    """
    Return the three loss components (R-yield term, W-data term, prior term)
    and basic stats for g=f+Lbinned (for diagnostics).
    """
    oi, ei, eiR, fX, nu, nuR, nu0, sig = outs
    log_ratio = torch.log(ei/eiR)
    Lbinned   = torch.sum(oi * log_ratio, dim=1)
    g = fX + Lbinned
    isR = (y==0)
    isW = (y==1)
    R_term = torch.sum((1.0-y)*w_e*(torch.exp(g)-1.0)).item()
    W_term = torch.sum(- y * g).item()
    Laux   = -0.5*torch.sum(((nu-nu0)**2 - (nuR-nu0)**2)/(sig**2)).item()
    stats = dict(
        g_all_min=float(g.min()), g_all_max=float(g.max()),
        g_R_mean=float(g[isR].mean()) if isR.any() else float('nan'),
        g_W_mean=float(g[isW].mean()) if isW.any() else float('nan'),
        Lb_mean=float(Lbinned.mean()), f_mean=float(fX.mean())
    )
    return R_term, W_term, Laux, stats

@torch.no_grad()
def compute_delta_T_and_terms(xW_bin: torch.Tensor,
                              xR_bin: torch.Tensor,
                              model: NPLM_WithNuis,
                              w_e_scalar: float,
                              grid_span_sig=4.0, grid_points=31):
    """
    Δ = max_ν 2*[  counts_W · log(e(ν)/e(νR))
                   - (N(R_ν) - N(R_0))
                   + log L(ν|A) - log L(0|A) ]
    where  N(R_ν) - N(R_0) ≈ w_e * Σ_R [exp(Lbinned_R(ν)) - 1].
    Return Delta and a dict of the maximizing ν and term breakdown.
    """
    oiW = model.oi(xW_bin)               # [N_W, nbins]
    countsW = oiW.sum(dim=0)             # [nbins]
    oiR = model.oi(xR_bin)               # [N_R, nbins]

    nu0, nuR, sig = model.nu0, model.nuR, model.sig
    eiR = model.eiR(nuR)                 # [nbins]

    axes = [torch.linspace(nu0[i]-grid_span_sig*sig[i],
                           nu0[i]+grid_span_sig*sig[i],
                           grid_points, dtype=torch.float32, device=xW_bin.device)
            for i in range(nu0.numel())]
    g = torch.meshgrid(*axes, indexing='ij')
    grid = torch.stack([t.reshape(-1) for t in g], dim=1)  # [G, n_nuis]

    Dmax = -torch.inf
    best = dict(nu=None, data=0.0, yield_=0.0, prior=0.0)
    for k in range(grid.shape[0]):
        nu = grid[k]
        ei  = model.ei(nu)                               # [nbins]
        log_ratio = torch.log(ei/eiR)                    # [nbins]

        data_term = torch.dot(countsW, log_ratio)        # Σ over W-bins
        Lb_R = torch.sum(oiR * log_ratio, dim=1)         # [N_R]
        yield_diff = w_e_scalar * torch.sum(torch.exp(Lb_R) - 1.0)  # N(Rν)-N(R0)
        prior_term = -0.5*torch.sum(((nu-nu0)/sig)**2)

        D = 2.0 * (data_term - yield_diff + prior_term)
        if D > Dmax:
            Dmax = D
            best['nu'] = nu.detach().cpu().numpy().tolist()
            best['data'] = float(data_term)
            best['yield_'] = float(yield_diff)
            best['prior'] = float(prior_term)

    Delta = float(torch.clamp(Dmax, min=0.0))
    return Delta, best

# =============== Sampling helpers ===============
def sample_pool(pool: np.ndarray, n: int, replace: bool) -> np.ndarray:
    """Sample n rows from a pool, with or without replacement."""
    if n <= 0:
        return np.empty((0, pool.shape[1]), dtype=np.float32)
    if replace:
        idx = rng.integers(0, len(pool), size=n)
    else:
        if n > len(pool):
            raise RuntimeError(f"Sampling without replacement needs {n}, but pool has only {len(pool)} rows left.")
        idx = rng.choice(len(pool), size=n, replace=False)
    return pool[idx]

def generate_W(lambda_val: float, replace: bool=True):
    """Generate a Poisson total experimental sample with Binomial split into signal and background."""
    M = rng.poisson(N_R0)
    if M == 0:
        X_W = np.empty((0, INPUT_DIM), dtype=np.float32)
    else:
        n_sig = rng.binomial(M, lambda_val)
        n_bkg = M - n_sig
        Xs = sample_pool(sig_pool, n_sig, replace)
        Xb = sample_pool(bkg_pool, n_bkg, replace)
        X_W = np.vstack([Xs, Xb]).astype(np.float32)
        perm = rng.permutation(len(X_W))
        X_W = X_W[perm]
    y_W = np.ones((X_W.shape[0],), dtype=np.float32)
    return torch.from_numpy(X_W).to(device), torch.from_numpy(y_W).to(device)

# =============== Weight clipping stats ===============
def weight_clip_stats(model: nn.Module, clip_val: float):
    """Count how many parameters are at ±clip (diagnostic only)."""
    total = 0; at_pos = 0; at_neg = 0
    for p in model.parameters():
        if p.requires_grad and p.data.numel() > 0:
            d = p.data
            total += d.numel()
            at_pos += (d == clip_val).sum().item()
            at_neg += (d == -clip_val).sum().item()
    frac_pos = at_pos / total if total else 0.0
    frac_neg = at_neg / total if total else 0.0
    return dict(total=total, at_pos=at_pos, at_neg=at_neg,
                frac_pos=frac_pos, frac_neg=frac_neg)

# =============== Train & score one toy (with nuisance) ===============
def train_and_score_one_toy_with_nuis(lambda_val: float,
                                      edges: torch.Tensor,
                                      A0: torch.Tensor, A1: torch.Tensor,
                                      nu0_init: torch.Tensor,
                                      sigma_vec: torch.Tensor,
                                      nu_ref: torch.Tensor):
    # Generate W
    X_W, y_W = generate_W(lambda_val, replace=USE_REPLACE_SAMPLING)
    X_Z = torch.cat([X_R, X_W], dim=0)
    y_Z = torch.cat([
        torch.zeros((X_R.shape[0],), dtype=torch.float32, device=device),
        y_W
    ], dim=0)

    # 1D projection used for binning
    idx_bin = feature_cols.index(BIN_VAR)
    xZ_bin = X_Z[:, idx_bin].float()
    xR_bin = X_R[:, idx_bin].float()
    xW_bin = X_W[:, idx_bin].float()

    # Initialize ν at prior mean
    nu_init = nu0_init.clone().detach()

    model = NPLM_WithNuis(edges, A0, A1, nu_init, nu_ref, nu0_init, sigma_vec, WCLIP).to(device)
    opt = optim.Adam([{'params': model.f.parameters()},
                      {'params': [model.nu]}], lr=LR)

    # Training loop
    last_loss = None
    for epoch in range(1, EPOCHS+1):
        opt.zero_grad(set_to_none=True)
        outs = model(X_Z, xZ_bin)
        loss = npl_loss_with_nuis(y_Z, w_e_scalar, outs)
        loss.backward()
        opt.step()
        model.f.clip_weights()

        if epoch % PRINT_EVERY_EPOCH == 0 or epoch == 1:
            clip_stat = weight_clip_stats(model.f, WCLIP)
            print(f"[λ={lambda_val:>4.2f}] Epoch {epoch:6d}/{EPOCHS} "
                  f"loss={loss.item():.6f} nu={model.nu.detach().cpu().numpy()} "
                  f"| clip_pos={clip_stat['frac_pos']:.3%}, clip_neg={clip_stat['frac_neg']:.3%}")

        if epoch % LOG_LOSS_COMPONENTS_EVERY == 0:
            R_term, W_term, Laux, stats = decompose_loss_terms(y_Z, w_e_scalar, outs)
            print(f"    ├─ loss terms: R_term={R_term:.3f}, W_term={W_term:.3f}, -Laux={-Laux:.3f} "
                  f"(so +2*Laux contributes to tau).")
            print(f"    └─ g=f+Lb stats: "
                  f"min={stats['g_all_min']:.3f}, max={stats['g_all_max']:.3f}, "
                  f"g_R_mean={stats['g_R_mean']:.3f}, g_W_mean={stats['g_W_mean']:.3f}")

        # simple divergence guard
        if not math.isfinite(loss.item()):
            print("⚠️ Loss is non-finite; aborting early.")
            break
        last_loss = loss.item()

    with torch.no_grad():
        outs = model(X_Z, xZ_bin)
        final_loss = npl_loss_with_nuis(y_Z, w_e_scalar, outs).item()
        tau = -2.0 * final_loss
        R_term, W_term, Laux, stats = decompose_loss_terms(y_Z, w_e_scalar, outs)
        tau_components = dict(
            two_W =  2.0 * (-W_term),               # 2 * sum_W(g)
            two_R = -2.0 * (R_term),                # -2 * sum_R w_e (exp(g)-1)
            two_Laux = 2.0 * (Laux),                # +2 * Laux
        )

    # Compute Delta via grid scan over ν, including data/yield/prior parts
    Delta, best = compute_delta_T_and_terms(xW_bin, xR_bin, model, w_e_scalar,
                                            grid_span_sig=4.0, grid_points=31)
    T = tau - Delta

    # Detailed decomposition prints
    print("    [TAU decomposition]")
    print(f"      2*Σ_W g = {tau_components['two_W']:.3f}, "
          f"-2*Σ_R w_e(exp(g)-1) = {tau_components['two_R']:.3f}, "
          f"+2*Laux = {tau_components['two_Laux']:.3f}  → tau = {tau:.3f}")
    print("    [DELTA decomposition (at best ν)]")
    print(f"      data_term = {best['data']:.3f}, yield_term = {best['yield_']:.3f}, prior_term = {best['prior']:.3f}, "
          f"→ Delta = {Delta:.3f}, best_nu = {np.array(best['nu'])}")
    if Delta > 5.0 * max(1.0, abs(tau)):
        print("⚠️ WARNING: Δ is much larger than τ; check bin-fit (A0/A1), prior σ, or data scale.")
    if any(np.isnan(v) or np.isinf(v) for v in [tau, Delta, T]):
        print("⚠️ WARNING: NaN/Inf detected in tau/Delta/T.")

    return tau, Delta, T, dict(
        tau_components=tau_components,
        delta_components=best,
        nu_final=model.nu.detach().cpu().numpy()
    )

# =============== Prepare bins & bin-fit on R ===============
edges = build_bins_from_array(X_R_np[:, feature_cols.index(BIN_VAR)], nbins=30, margin=0.05)
A0, A1, hist_R = build_binfit_from_R(X_R_np[:, feature_cols.index(BIN_VAR)], edges,
                                     n_nuis=2, a0_scale=0.12, a1_std=0.01)
NU_REF   = torch.zeros(2, dtype=torch.float32, device=device)          # reference ν_R
NU0_MEAN = torch.tensor([0.02, 0.00], dtype=torch.float32, device=device) # prior mean
SIGMA    = torch.tensor([0.05, 0.05], dtype=torch.float32, device=device) # prior σ

# Print binning & bin-fit diagnostics
lo, hi = edges[0].item(), edges[-1].item()
print(f"Bins on {BIN_VAR}: {edges.numel()-1} bins, "
      f"[{lo:.3g},{hi:.3g}] ; R histogram sum={hist_R.sum()}, "
      f"min/max per-bin counts={hist_R.min()}/{hist_R.max()}")
print(f"A0 stats: min={A0.min().item():.4g}, max={A0.max().item():.4g}, mean={A0.mean().item():.4g}")
A1_np = A1.detach().cpu().numpy()
print(f"A1 row0 L2={np.linalg.norm(A1_np[0]):.4g}, row1 L2={np.linalg.norm(A1_np[1]):.4g} (small = weak sensitivity)")
if (A0 <= 1e-6).any():
    print("⚠️ WARNING: some A0 bins are at epsilon (clamped). Consider increasing a0_scale.")

# =============== Main loop ===============
def main():
    print(f"\nN_R = {N_R}, N_R0 = {N_R0}, w_e = {w_e_scalar:.6f}, DOF = {DF_FIXED}")
    print(f"Training EPOCHS={EPOCHS}, LR={LR}, clip={WCLIP}, PRINT_EVERY={PRINT_EVERY_EPOCH}")
    results = {}

    for lam in LAMBDAS:
        print(f"\n--- λ = {lam:.2f} ---")
        taus, deltas, Ts = [], [], []
        extra_logs = []
        for i in range(1, NUM_TOYS+1):
            tau, Delta, T, log_pack = train_and_score_one_toy_with_nuis(
                lam, edges, A0, A1, NU0_MEAN, SIGMA, NU_REF
            )
            taus.append(tau); deltas.append(Delta); Ts.append(T); extra_logs.append(log_pack)
            print(f"  Toy {i:02d}/{NUM_TOYS}: tau={tau:.3f}, Δ={Delta:.3f}, T={T:.3f}")

        taus = np.asarray(taus); deltas = np.asarray(deltas); Ts = np.asarray(Ts)
        print(f"  Summary(λ={lam:.2f}): "
              f"mean(tau)={taus.mean():.3f}, mean(Δ)={deltas.mean():.3f}, mean(T)={Ts.mean():.3f}, "
              f"min/max T = {Ts.min():.3f}/{Ts.max():.3f}")
        results[lam] = {"tau": taus, "Delta": deltas, "T": Ts, "logs": extra_logs}

    # KS on λ=0 (uniformity of p-values)
    if 0.0 in results:
        Ts0 = results[0.0]["T"]
        pvals0 = 1.0 - chi2.cdf(Ts0, df=DF_FIXED)
        ks_stat, ks_p = kstest(pvals0, 'uniform')
        print(f"\n[λ=0] KS vs U(0,1): stat = {ks_stat:.4f}, p = {ks_p:.4f}")
        if ks_p < 0.05:
            print("⚠️ p-value CDF under λ=0 deviates from Uniform(0,1). Check model DOF, bin-fit, or priors.")

    return results

if __name__ == "__main__":
    results = main()

    # ===== Save JSON =====
    out = {
        "meta": {
            "seed": SEED, "device": str(device),
            "epochs": EPOCHS, "lr": LR,
            "num_toys": NUM_TOYS, "lambdas": LAMBDAS,
            "input_dim": INPUT_DIM, "hidden_dim": [H1,H2], "output_dim": 1,
            "df_fixed": DF_FIXED, "input_csv": INPUT_CSV,
            "N_R": N_R, "N_R0": N_R0, "clip_value": WCLIP,
            "feature_cols": feature_cols, "bin_var": BIN_VAR,
            "nuis_prior_mean": [float(x) for x in NU0_MEAN.cpu().numpy()],
            "nuis_prior_sigma": [float(x) for x in SIGMA.cpu().numpy()],
            "n_nuis": 2, "nbins": int(edges.numel()-1),
            "loss": "sum[(1-y) w_e (exp(f+Lb)-1) - y(f+Lb)] - 0.5*((nu-nu0)^2-(nuR-nu0)^2)/sig^2",
            "delta": "grid scan on nu with data term, yield term, Gaussian prior"
        },
        "results": {}
    }
    for lam in results:
        out["results"][str(lam)] = {
            "tau": results[lam]["tau"].tolist(),
            "Delta": results[lam]["Delta"].tolist(),
            "T": results[lam]["T"].tolist(),
            "logs": results[lam]["logs"]  # per-toy breakdown and final ν
        }

    with open("T_tau_Delta_WITH_NUIS_VERBOSE.json","w",encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("✅ Saved to T_tau_Delta_WITH_NUIS_VERBOSE.json")

    # ===== Optional: visualize p-value CDFs (kept unchanged) =====
    try:
        colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
        plt.figure(figsize=(7,5))
        for i, lam in enumerate(sorted(results.keys())):
            Ts = results[lam]["T"]
            p = 1.0 - chi2.cdf(Ts, df=DF_FIXED)
            s = np.sort(p); y = np.arange(1, len(s)+1)/len(s)
            plt.step(s, y, where='post', lw=2.0, color=colors[i%len(colors)],
                     label=fr'$\lambda={lam}$')
        plt.plot([0,1],[0,1],'--',color='0.3',lw=1.2)
        plt.xlim(-0.02, 1.02); plt.ylim(-0.02, 1.02)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(1.0)
        plt.grid(True, ls=':', lw=0.8, alpha=0.5)
        plt.xlabel('p-value'); plt.ylabel('Empirical CDF')
        plt.title('LRT p-value CDF with Nuisance (verbose run)')
        plt.legend(loc='lower right', frameon=True, edgecolor='black')
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("Plotting skipped:", e)
