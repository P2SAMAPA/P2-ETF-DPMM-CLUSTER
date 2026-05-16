import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import config
import data_manager
from dpmm_model import DPMMVariational

def compute_etf_features(returns_df, window=60):
    """
    For each ETF, compute summary statistics over the last `window` days:
    - mean return
    - standard deviation
    - skewness
    - kurtosis
    Returns feature matrix (n_etfs, 4).
    """
    features = []
    for col in returns_df.columns:
        rets = returns_df[col].iloc[-window:].dropna()
        if len(rets) < 10:
            features.append([np.nan]*4)
            continue
        mean = rets.mean()
        std = rets.std()
        skew = rets.skew()
        kurt = rets.kurt()
        features.append([mean, std, skew, kurt])
    return np.array(features)

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (DPMM Clustering) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.OBSERVATION_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Compute features
        X_raw = compute_etf_features(returns, window=config.OBSERVATION_WINDOW)
        # Remove rows with NaN
        valid_idx = ~np.isnan(X_raw).any(axis=1)
        X = X_raw[valid_idx]
        valid_tickers = [tickers[i] for i in range(len(tickers)) if valid_idx[i]]
        if len(X) < 3:
            print("  Not enough valid ETFs")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Standardise features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit DPMM
        dpmm = DPMMVariational(n_components=config.N_COMPONENTS, alpha=config.ALPHA,
                               max_iter=config.MAX_ITER, tol=config.TOL)
        dpmm.fit(X_scaled)
        # Membership probabilities
        r = dpmm.predict_proba(X_scaled)   # (n_etfs, n_components)
        # Component means (in original scale)
        cluster_means = dpmm.get_cluster_means()
        # Inverse transform to original scale
        cluster_means_orig = scaler.inverse_transform(cluster_means)
        # The "best" cluster is the one with highest mean return (first feature)
        best_cluster = np.argmax(cluster_means_orig[:, 0])
        # For each ETF, its probability of belonging to the best cluster
        prob_best = r[:, best_cluster]
        # Rank ETFs by prob_best descending
        sorted_idx = np.argsort(prob_best)[::-1]
        top_etfs = []
        full_scores = {}
        for i, idx in enumerate(sorted_idx[:config.TOP_N]):
            ticker = valid_tickers[idx]
            score = prob_best[idx]
            top_etfs.append({"ticker": ticker, "prob_best_cluster": float(score)})
            full_scores[ticker] = float(score)
        print(f"  Top 3 ETFs by probability in high‑mean cluster: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "n_clusters": len(np.unique(np.argmax(r, axis=1))),
            "best_cluster_mean_return": float(cluster_means_orig[best_cluster, 0]),
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/dpmm_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Non‑Parametric Bayes ETF Clustering complete ===")

if __name__ == "__main__":
    main()
