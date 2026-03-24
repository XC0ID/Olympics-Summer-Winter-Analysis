# Results

## Regression model
| Metric | Value |
|--------|-------|
| RMSE   | see results/metrics/metrics.json |
| MAE    | see results/metrics/metrics.json |
| R2     | see results/metrics/metrics.json |

## Classification model
| Metric   | Value |
|----------|-------|
| Accuracy | 0.32  |

## Clustering model
| Metric          | Value |
|-----------------|-------|
| Best k          | 2     |
| Silhouette score| 0.74  |

## Output files
| File | Description |
|------|-------------|
| results/metrics/metrics.json | All model scores |
| results/metrics/country_clusters.csv | Cluster per country |
| results/visualizations/medal_trends.png | Top 10 medal trends |
| results/visualizations/regression_feature_importance.png | Feature importance |
| results/visualizations/actual_vs_pred.png | Actual vs predicted |
| results/visualizations/elbow_silhouette.png | Optimal k chart |
| results/visualizations/cluster_heatmap.png | Cluster profiles |