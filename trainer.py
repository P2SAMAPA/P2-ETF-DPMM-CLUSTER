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
    Returns feature matrix (n_etfs, 4) and list of valid tickers.
    """
    features = []
    tickers_kept = []
    for col in returns_df.columns:
        rets = returns_df[col].iloc[-window:].dropna()
        if len(rets) < 10:
            continue
        mean = rets.mean()
        std = rets.std()
        skew = rets.skew()
        kurt = rets.kurt()
        features.append([mean, std, skew, kurt])
        tickers_kept.append(col)
    return np.array(features), tickers_kept

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
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            # Use last `win` days of returns
            returns_win = returns.iloc[-win:]
            # Compute features on the last `win` days using a 60‑day window? No, the feature window should be relative to the overall window.
            # The DPMM expects features computed on the full `win` period (or a rolling window within it). To keep it simple,
            # we compute features directly on the `win` days (i.e., we treat each ETF's return series over the `win` days as a sample).
            # But we need a single feature vector per ETF. So we compute the mean, std, skew, kurt of the ETF's returns over the entire `win` days.
            # That's fine – it's a static feature for the window.
            features, valid_tickers = compute_etf_features(returns_win, window=win)  # use the window as the feature window
            if len(features) < 3:
                continue
            # Standardise features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)
            # Fit DPMM
            dpmm = DPMMVariational(n_components=config.N_COMPONENTS, alpha=config.ALPHA,
                                   max_iter=config.MAX_ITER, tol=config.TOL)
            dpmm.fit(X_scaled)
            r = dpmm.predict_proba(X_scaled)          # membership probabilities
            cluster_means = dpmm.get_cluster_means()
            # Inverse transform to original scale
            cluster_means_orig = scaler.inverse_transform(cluster_means)
            # Best cluster = highest mean return (first feature)
            best_cluster = np.argmax(cluster_means_orig[:, 0])
            prob_best = r[:, best_cluster]
            etf_scores = {valid_tickers[i]: prob_best[i] for i in range(len(valid_tickers))}
            window_results[win] = etf_scores
            for etf, score in etf_scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            # Fallback: use historical mean return (positive if >0, else small positive)
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        # Store full scores for all ETFs
        full_scores = {ticker: {"score": score, "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "prob_best_cluster": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by probability in best (highest‑return) cluster: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/dpmm_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== DPMM Clustering Engine (multi‑window) complete ===")

if __name__ == "__main__":
    main()
