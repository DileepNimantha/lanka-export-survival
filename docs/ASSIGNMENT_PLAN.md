# Advanced ML Assignment — Handoff & Plan

Context carried over from a planning session in Claude (Cowork), 26 Oct 2026.

## Source documents (in this docs/ folder)
- `ML_Assignment_Guidelines_MSc_AI.pdf` — **the official requirement. Only this counts.**
- `MLOps_Workshop_Sections_1-8_Commands.pdf` — step-by-step commands for this workshop repo (reference).

## Official marking scheme (from the guidelines PDF)
| # | Criterion | Marks |
|---|---|---|
| 1 | Problem definition & dataset: problem + relevance, data source, features/target, size, preprocessing, ethical use | 15 |
| 2 | New ML algorithm **not covered in lectures**, no deep learning; justify choice and how it differs from DT / LR / k-NN | 15 |
| 3 | Training & evaluation: train/val/test split, hyperparameters, metrics, results with tables/plots | 20 |
| 4 | Explainability: at least one of SHAP / LIME / feature importance / PDP; what the model learned, key features, alignment with domain knowledge | 20 |
| 5 | Critical discussion: limitations, data quality, bias/fairness, real-world impact, ethics | 10 |
| 6 | Report quality & technical clarity | 10 |
| 7 | **Bonus**: front end where users input data and see predictions/explanations | 10 |

No image-processing application. Must collect/compile a **local** dataset.

**Deliverables (Moodle):** report PDF · source code (ZIP/GitHub) incl. preprocessing, training, evaluation, explainability scripts · dataset or description of how obtained · 3–5 min demo video of the front end.

## Decided approach
Build the assignment as an end-to-end MLOps project with the same architecture as this
IT5232 workshop repo (slide: "The system we will build"):
Codespaces → GitHub PR → GitHub Actions CI + quality gate → Docker image to GHCR →
approval + OIDC → Azure Container Apps (canary, rollback) → App Insights / Log Analytics →
Evidently drift → retraining opens a new PR. DagsHub for MLflow tracking/registry + DVC data.

**Reuse the architecture, not the case.** Bank Marketing (Portuguese UCI data) + CatBoost is
the workshop's own case — the assignment needs our own local problem, dataset and algorithm
(reusing them across modules is a plagiarism risk).

MLOps only earns the 10-mark bonus; the 90 core marks come from sections 1–6, so ML work comes first.

## Known issues in this workshop repo
- `main` lacks `Makefile`, `pyproject.toml`, `uv.lock`, `src/config.py`, `.dvc/config`
  (instructor's s2, s4–s8 branches lack them too). They exist on `origin/s1` and `origin/s3`:
  `git checkout origin/s3 -- Makefile pyproject.toml uv.lock src/config.py .dvc/config`
- CatBoost is hard-coded in train.py, evaluate.py, gate.py, registry (mlflow.catboost),
  app/main.py, tests and the `.cbm` artifact → make model-agnostic (MLflow pyfunc or sklearn pipeline + joblib).
- `cd.yml` still contains temporary DagsHub debug steps — remove.
- Setup gotchas: `DAGSHUB_OWNER` must be an Actions **variable** (else CI silently skips);
  `DAGSHUB_REPO` must match the DagsHub repo name; publish image via CD and make it **public**
  before `bootstrap.sh`; OIDC trust is per repo → rerun bootstrap for the new repo;
  create `production` environment with required reviewer.

## Build order (maps to workshop sections)
| Workshop | Assignment work | Criterion |
|---|---|---|
| S1 Environment | New repo from this template, restore missing files, `make doctor` | – |
| S2 Data + experiments | Local dataset, new Pandera/Pydantic schema, **train/val/test** split, baselines (e.g. LR, RF, XGBoost) vs **new algorithm**, tuning (e.g. Optuna) logged to MLflow | 1, 2, 3 |
| S3 DVC pipeline | Same stages + new `explain` stage (SHAP global+local, PDP and/or LIME) | 3, 4 |
| S4 Gate + registry | Reuse; pick gate metric for the problem | – |
| S5 API + Docker | Add `/explain` endpoint + **Streamlit UI** (prediction + explanation) | 7 |
| S6 Azure CI/CD | Rename resources, rerun bootstrap for new repo | 7 |
| S7 Monitoring | Update `replayer/` for new features & drift scenarios | 5 (material) |
| S8 Retraining | Reuse | – |
| Report + video | Report structured by marking criteria + architecture diagram; demo: predict+explain → PR with gate report → approval → canary → drift → retrain PR | 6 |

## Decided: problem + dataset (27 Oct 2026)
**Sri Lanka export survival.** Predict whether a Sri Lankan export flow (HS6 product →
destination country) present in year t is still exported in year t+1.
- Data: CEPII BACI, HS12 version V202601 (2012–2024), one zip of 1.27 GB, Etalab 2.0 licence:
  `https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS12_V202601.zip`.
  `src/download.py` downloads it once and keeps only exporter 144 (Sri Lanka) plus each
  destination market's size; the small extract is versioned with DVC and is what gets submitted.
- Optional extra features: CEPII Gravity (distance, language, trade agreements, GDP).
- Time split by year; 2020 (COVID) and 2022 (economic crisis) give real drift.
- Ruled out: weather (too common), electricity (PDF scraping), WFP food prices (did not beat
  the naive baseline), tourist reviews (used in a previous assignment), cricket (weaker
  real-world relevance).

## Decided: new algorithm (27 Oct 2026)
**Explainable Boosting Machine** (`interpret`), a glass-box GA2M: one learned shape function per
feature plus 10 pairwise interaction terms, fitted by cyclic gradient boosting and bagging.
Chosen because it is interpretable by design (criterion 4) while competitive with black-box
boosting. Replaces CatBoost everywhere (train, evaluate, gate, API, Docker, CD); the model is
saved as `models/model.joblib` and logged with `mlflow.sklearn` (skops, EBM type trusted).
Baselines in `src/experiments.py`: new-flow rule, logistic regression, decision tree, random
forest, histogram gradient boosting. Still to confirm with the lecturer that EBM was not covered.
