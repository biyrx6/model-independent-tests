import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

# ----------------------------
# Parameter setup
# ----------------------------
np.random.seed(718)
d = 5
m1, m2 = 5000, 8000   # background train / test sizes
n1, n2 = 5000, 8000   # experimental (mixture) train / test sizes

# Mixture: signal ~ N(0.5,1), background ~ N(0,1)
lambda_list = [0, 0.01, 0.03, 0.05, 0.1, 1]
repeats = 200

# Storage for p-values
results = {lam: {"lrt": [], "auc": [], "mce": []} for lam in lambda_list}

def compute_log_psi(probs, pi):
    """log ψ̂(p) = log(((1-π)/π) * p/(1-p))."""
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    psi = (1 - pi) / pi * probs / (1 - probs)
    return np.log(psi)

# ----------------------------
# Simulation loop (independent sampling)
# ----------------------------
for lam in lambda_list:
    for rep in range(repeats):
        # ----- Train set sampling -----
        X1 = np.random.normal(0, 1, size=(m1, d))
        n1_s = np.random.binomial(n1, lam)
        n1_b = n1 - n1_s
        W1_sig = np.random.normal(0.5, 1, size=(n1_s, d))
        W1_bg  = np.random.normal(0.0, 1, size=(n1_b, d))
        W1 = np.vstack([W1_sig, W1_bg])

        y_back = np.zeros(m1)
        y_exp_train = np.ones(n1)

        # ----- Test set sampling -----
        X2 = np.random.normal(0, 1, size=(m2, d))
        n2_s = np.random.binomial(n2, lam)
        n2_b = n2 - n2_s
        W2_sig = np.random.normal(0.5, 1, size=(n2_s, d))
        W2_bg  = np.random.normal(0.0, 1, size=(n2_b, d))
        W2 = np.vstack([W2_sig, W2_bg])

        # ----- Train classifier -----
        X_train = np.vstack([X1, W1])
        y_train = np.hstack([y_back, y_exp_train])
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_leaf=50,
            random_state=rep,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # Predicted probabilities on test sets
        pX2 = clf.predict_proba(X2)[:, 1]
        pW2 = clf.predict_proba(W2)[:, 1]

        # ===== LRT p-value =====
        pi = n1 / (n1 + m1)
        log_psi_W2 = compute_log_psi(pW2, pi)
        log_psi_X2 = compute_log_psi(pX2, pi)
        T_tilde = np.mean(log_psi_W2)
        T0 = np.mean(log_psi_X2)
        sigma0_sq = np.var(log_psi_X2, ddof=1)
        Z_lrt = np.sqrt(n2) * (T_tilde - T0) / np.sqrt(2 * sigma0_sq)
        p_val_lrt = 1 - norm.cdf(Z_lrt)
        results[lam]["lrt"].append(p_val_lrt)

        # ===== AUC p-value =====
        labels_auc = np.concatenate([np.zeros(len(pX2)), np.ones(len(pW2))])
        scores_auc = np.concatenate([pX2, pW2])
        theta_hat = roc_auc_score(labels_auc, scores_auc)
        # Null var for AUC under H0: theta=0.5 (Mann–Whitney)
        var_auc_null = (m2 + n2 + 1) / (12.0 * m2 * n2)
        Z_auc = (theta_hat - 0.5) / np.sqrt(var_auc_null)
        p_val_auc = 1 - norm.cdf(Z_auc)
        results[lam]["auc"].append(p_val_auc)

        # ===== MCE p-value =====
        pi_thresh = pi  # threshold = prior
        mce_fp = np.mean(pX2 > pi_thresh)      # false positive rate
        mce_fn = np.mean(pW2 < pi_thresh)      # false negative rate
        mce_hat = 0.5 * (mce_fp + mce_fn)

        theta_z_hat = (m2 * mce_fp + n2 * (1 - mce_fn)) / (m2 + n2)
        var_mce = 0.25 * theta_z_hat * (1 - theta_z_hat) * (1/m2 + 1/n2)
        Z_mce = (mce_hat - 0.5) / np.sqrt(var_mce)
        p_val_mce = norm.cdf(Z_mce)  
        results[lam]["mce"].append(p_val_mce)

    print(f"Completed lambda = {lam}")

# ----------------------------
# Build p-value table
# ----------------------------
records = []
for lam, stats in results.items():
    for i, (p_lrt, p_auc, p_mce) in enumerate(
        zip(stats["lrt"], stats["auc"], stats["mce"]), start=1
    ):
        records.append({
            "Lambda": lam,
            "Trial": i,
            "LRT p-value": p_lrt,
            "AUC p-value": p_auc,
            "MCE p-value": p_mce
        })

df = pd.DataFrame(records)
print("\nFirst 10 rows of p-value table:")
print(df.head(10).to_string(index=False))

# ----------------------------
# Show p-value tables as figures 
# ----------------------------
for key, col in [("lrt", "LRT p-value"), ("auc", "AUC p-value"), ("mce", "MCE p-value")]:
    sub = df[["Lambda", "Trial", col]].pivot(index="Trial", columns="Lambda", values=col)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    tbl = ax.table(cellText=np.round(sub.values, 4),
                   rowLabels=sub.index,
                   colLabels=sub.columns,
                   cellLoc='center',
                   loc='center')
    tbl.scale(1, 1.5)
    ax.set_title(f"{col} Table", pad=20)
    plt.tight_layout()
    plt.show()

# ----------------------------
# Plot empirical CDFs of p-values 
# ----------------------------
stat_map = {"lrt": "LRT", "auc": "AUC", "mce": "MCE"}

for key, title in stat_map.items():
    fig, ax = plt.subplots(figsize=(6, 4))
    for lam in lambda_list:
        vals = np.sort(results[lam][key])
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", label=f"λ={lam}")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"{title} p-value CDF")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.7)  
    plt.tight_layout()
    plt.show()

print("\nAll done! (figures popped up instead of being saved)")
