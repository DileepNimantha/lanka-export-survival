# Lanka export survival

Predicts whether a Sri Lankan export flow (one product to one destination country) will
still be exported the following year, and explains why. Built as an end-to-end MLOps
system: versioned data, tracked experiments, a model registry with a quality gate, a
FastAPI service on Azure Container Apps with canary releases, drift monitoring, and
automated retraining.

| Folder | What it holds |
|---|---|
| `src/` | data contract, training, evaluation, quality gate, model registry |
| `app/` | the prediction API |
| `monitoring/` | log export and drift checks |
| `replayer/` | simulated production traffic for the incident drill |
| `infra/` | Azure resources (Bicep) and the bootstrap script |
| `.github/workflows/` | CI, CD, monitoring and retraining |

Common commands: `make test`, `make repro`, `make experiments`, `make serve`.

Data: BACI international trade database (CEPII, Gaulier and Zignago, 2010), HS 2012
version, licensed Etalab 2.0. Only Sri Lanka's export flows are kept.

Pipeline structure adapted from the IT5232 MLOps workshop (bank marketing).
