# B-COSified GNN Experiments

This folder contains baseline-vs-B-COS experiment scripts with `readout_then_agg`.

## Scripts

- `gcn_ba2motif_readout_then_agg.py`
  - Binary BA2Motif experiment.
  - Runs `vanilla_gcn_mean` and `bcos_gcn_mean_readout_then_agg` across 3 seeds.
  - Exports per-seed and summary CSVs + best-seed completeness plot.

- `gcn_dihalo_readout_then_agg.py`
  - Multi-class Di-Halo Benzene experiment.
  - Runs `vanilla_gcn_mean` and `bcos_gcn_mean_readout_then_agg` across 3 seeds.
  - Exports per-seed and summary CSVs + best-seed completeness plot.

## Run

```bash
python3 shaique_updates/codes/bcosified_gnns/gcn_ba2motif_readout_then_agg.py
python3 shaique_updates/codes/bcosified_gnns/gcn_dihalo_readout_then_agg.py
```

## Outputs

Each script writes under:

- `shaique_updates/codes/bcosified_gnns/results/`
- `shaique_updates/codes/bcosified_gnns/results/completeness_plots/`
