# Non‑Parametric Bayes ETF Clustering

Dirichlet Process Mixture Model (DPMM) for ETF clustering. Automatically discovers the number of clusters from return distribution features (mean, std, skew, kurt). Outputs the probability that each ETF belongs to the best (highest‑return) cluster.

- **DPMM:** Stick‑breaking representation, variational inference
- **Features:** Rolling 60‑day mean, volatility, skewness, kurtosis
- **Output:** Top 3 ETFs by cluster membership probability
- **Dashboard:** Shows cluster count, best cluster mean return, and full ranking

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
