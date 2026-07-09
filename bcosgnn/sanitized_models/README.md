# Sanitized Model Modules

This folder contains sanitized, reusable model definitions extracted from experiment notebooks/scripts.

Current canonical implementation:
- `BCosGNN` + `BcosGINConv` + readout classes from the BA2 motifs notebook `ba2motifs_shaique_GIN_OG_CC.ipynb`.

## Import

```python
from bcosgnn.sanitized_models import (
    BCosGNN,
    BcosGINConv,
    AggThenReadout,
    ReadoutThenAgg,
)
```
