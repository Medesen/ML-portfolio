# ML Portfolio - Churn Prediction

> **📍 Project Location**: This project lives in `ML-portfolio/end2end_pipe/`  
> **⚠️ Important**: All commands below should be run from the `end2end_pipe/` directory:
> ```bash
> cd end2end_pipe
> ```

A small churn prediction pipeline using scikit-learn with a reproducible training flow, schema alignment at inference, experiment artifacts, CI tests, and dev tooling.

### Key Features & Recent Improvements

**Production-Ready ML Pipeline:**
- ✅ **Automated Drift Detection**: Real-time monitoring with PSI (Population Stability Index) for categorical features, statistical tests for numerical features, and prediction distribution tracking
- ✅ **Model Caching**: Optimized service performance - model loaded once at startup instead of per-request
- ✅ **Schema Alignment**: Automatic feature schema validation and alignment at inference (adds missing columns, drops extras, reorders)
- ✅ **Input Validation**: Request size limits and validation (configurable via `MAX_PREDICTION_RECORDS`)
- ✅ **Comprehensive Testing**: 43+ tests covering schema alignment, drift detection, threshold tuning, and service validation
- ✅ **Model Card**: Complete model documentation following best practices (intended use, limitations, bias considerations, ethical guidelines)
- ✅ **License**: MIT License for open-source portfolio use

**MLOps Best Practices:**
- Versioned artifacts with run IDs, git SHAs, and data hashes
- Optional MLflow experiment tracking and model registry
- FastAPI service with Prometheus metrics
- Docker/Docker Compose deployment
- CI/CD with GitHub Actions
- Pre-commit hooks (black, isort, flake8)
- Reproducible builds with pinned dependencies

### Quickstart

1) Create and activate a virtualenv, then install deps:
```
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

Alternatively, install runtime dependencies directly from the package metadata:
```
pip install .
```

Reproducible installs (pin to known-good versions):
```
pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt
pip install -e . -c constraints.txt
```
Note: `[project.dependencies]` in `pyproject.toml` is the source of truth for runtime dependencies. `constraints.txt` pins versions for reproducible builds. `requirements*.txt` are provided for familiarity and IDE support, but package installs should come from `pyproject.toml`.

If you plan to enable MLflow tracking (`--mlflow`), install the MLflow extra:
```
pip install -e .[mlflow]
```

2) Run training (from repo root):
```
python -m ml_portfolio_churn.main train --pos-label Yes --config config.sample.json
```

3) Run prediction:
```
python -m ml_portfolio_churn.main predict --input data/new_data.csv --no-pii --output predictions.csv
```

4) CLI help
```
python -m ml_portfolio_churn --help
```

### Docker (reproducible runs)

Build the image:
```
docker build -t churn:latest .
```

Train (mount local `data/` and `models/` to persist):
```
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
churn:latest churn train --pos-label Yes --config /app/config.sample.json
```

Predict:
```
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD:/out" \
churn:latest churn predict --input /app/data/new_data.csv --no-pii --output /out/predictions.csv
```

### Configuration

Edit a copy of `config.sample.json` and pass it via `--config`.

Keys:
- `random_state`: seed for reproducibility
- `splits.test_size`: fraction for held-out test set
- `splits.val_size`: fraction of train for validation
- `grid_search.cv`: CV folds
- `grid_search.scoring`: metric (default ROC AUC)
- `grid_search.param_grid`: RF params to search

### Artifacts (saved in `models/` per run)
- Model: `churn_model.joblib` and `churn_model_{run_id}.joblib`
- Diagnostics: `model_diagnostics_{run_id}.txt`
- Plots: `confusion_matrix_val_{run_id}.png`, `roc_curve_val_{run_id}.png`, `feature_importance_{run_id}.png`, `confusion_matrix_test_{run_id}.png`, `roc_curve_test_{run_id}.png`
- Metadata: `model_metadata.json` (schema, git SHA, data hash, artifact paths)
- Environment/config: `requirements_{run_id}.txt`, `run_config_{run_id}.json`

### Dev tooling
- Tests: `pytest -q`
- Pre-commit hooks (format + lint):
```
pre-commit install
pre-commit run --all-files
```
- Make targets: `make help` lists build/up/down/logs/health/metrics/train-sample. You can override variables, e.g. `make health TOKEN=yourtoken`.

### CLI install (optional)

Install as a CLI locally (editable mode):
```
pip install -e .
```
Then run:
```
churn train --pos-label Yes --config config.sample.json
churn predict --input data/new_data.csv --no-pii
```

### Notes
- **Dependencies**: All runtime dependencies properly declared in `pyproject.toml`. MLflow is optional (`pip install -e .[mlflow]`). Pydantic and Pandera included for validation.
- **Testing**: 43+ comprehensive tests covering schema alignment, drift detection, threshold tuning, service validation, and edge cases. Run with `pytest -q`.
- **Performance**: Service uses model caching (loaded at startup) for optimal inference speed. No disk I/O per request.
- **ARFF to CSV**: `src/ml_portfolio_churn/convert_arff_to_csv.py` (robust parser)
- **Positive class handling**: Explicit label encoding; probabilities are aligned to class `1`.
- **Data validation**: Warns on >70% missing columns, constant columns; at predict-time, missing expected columns are added as NA and extras dropped.
- **Pandera validation** (inference): Input validated/coerced with permissive schema. Extend `ml_portfolio_churn/features.py` to specify expected columns/types or enable strict mode.
- **Threshold tuning**: Validation-tuned decision threshold saved to metadata; choose objective with `--threshold-objective` (`f1`, `precision_constrained_recall` with `--min-precision`, or `top_k` with `--top-k-ratio`).
- **ARFF ingestion**: Prefers `liac-arff`; falls back to manual parser for quoted attributes and values.
- **CHURN_ROOT**: Set `CHURN_ROOT=/path/to/project` to change where `data/`, `models/`, logs, and outputs are read/written.
- **Predictions**: By default, saved with timestamped filename to avoid overwriting; use `--output` to control the path.
- **Drift Detection**: Reference statistics computed during training and stored in metadata. Use `POST /drift` endpoint to analyze new data batches.

### MLflow (optional experiment tracking)

If you haven't already, install the MLflow extra:
```
pip install -e .[mlflow]
```

Enable local MLflow tracking (logs under `mlruns/`):
```
python -m ml_portfolio_churn.main train --pos-label Yes --config config.sample.json --mlflow
```

Or point to a tracking server:
```
python -m ml_portfolio_churn.main train --pos-label Yes --config config.sample.json --mlflow \
  --mlflow-uri http://your-mlflow:5000
```

### Minimal service (optional)

Run the FastAPI service (install extra first):
```
pip install -e .[service]
uvicorn ml_portfolio_churn.service:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`: basic model/metadata availability.
- `GET /drift`: returns training schema, baseline statistics, and drift detection info.
- `POST /drift`: analyzes drift on batch data (body: `{ "records": [...] }`). Returns drift metrics for numerical features (mean/std changes), categorical features (PSI), and prediction distribution.
- `POST /predict`: body `{ "records": [ {"feature": value, ...}, ... ] }` returns predictions and probabilities.

### Docker Compose (recommended for demo)

Start the service:
```
docker compose up --build -d
```

Configure:
- Set `SERVICE_TOKEN` env var in your shell to secure endpoints (Bearer token).
- Set `MAX_PREDICTION_RECORDS` to limit batch size (default: 1000).
- Set `DRIFT_THRESHOLD_NUMERICAL` for numerical drift sensitivity (default: 0.1 = 10%).
- Set `DRIFT_THRESHOLD_CATEGORICAL` for categorical drift PSI threshold (default: 0.1).
- Set `DRIFT_THRESHOLD_PREDICTION` for prediction distribution drift (default: 0.1 = 10%).
- Data and models are mounted from `./data` and `./models`.

Optional Prometheus:
- Uncomment the `prometheus` service in `docker-compose.yml` and provide `infra/prometheus.yml` (included) to scrape `http://churn-service:8000/metrics`.

If you haven't trained a model yet, run a quick local training:
```
make train-sample
```

### Sample request and curl

Example request body for `POST /predict` (replace fields with your feature names):
```json
{
  "records": [
    { "num": 1, "cat": "a" },
    { "num": 5, "cat": "b" }
  ]
}
```

Curl examples (set your `SERVICE_TOKEN`):
```bash
export SERVICE_TOKEN=changeme

# Health
curl -H "Authorization: Bearer $SERVICE_TOKEN" http://localhost:8000/health

# Get drift baseline info
curl -H "Authorization: Bearer $SERVICE_TOKEN" http://localhost:8000/drift

# Analyze drift on batch data
curl -H "Authorization: Bearer $SERVICE_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"records":[{"num":1,"cat":"a"},{"num":5,"cat":"b"}]}' \
     http://localhost:8000/drift

# Predict
curl -H "Authorization: Bearer $SERVICE_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"records":[{"num":1,"cat":"a"},{"num":5,"cat":"b"}]}' \
     http://localhost:8000/predict
```

### Security notes

- The `/metrics` endpoint is intentionally unauthenticated to allow Prometheus and container healthchecks to scrape it. Other endpoints require a static Bearer token when `SERVICE_TOKEN` is set.
- For production deployments, prefer stronger controls: mTLS or OIDC-backed auth, network policies (firewall/NSG), secrets management for tokens, and audit logging. Avoid including PII in logs or metrics.

Register model and set stage (requires MLflow Model Registry):
```
python -m ml_portfolio_churn.main train --pos-label Yes --config config.sample.json --mlflow \
  --mlflow-uri http://your-mlflow:5000 \
  --registered-model-name churn_rf \
  --model-stage Staging
```

---

## Model Card

### Model Overview
- **Model Type**: Random Forest Classifier
- **Task**: Binary classification (customer churn prediction)
- **Framework**: scikit-learn
- **Version**: See `model_metadata.json` for run-specific versions
- **Last Updated**: Model artifacts are timestamped per training run

### Intended Use
**Primary Use Cases**:
- Predicting customer churn probability for retention campaigns
- Educational demonstration of end-to-end ML pipeline best practices
- Portfolio showcase of MLOps capabilities

**Intended Users**:
- Data science teams evaluating customer retention strategies
- ML engineers learning production deployment patterns
- Technical recruiters assessing MLOps competency

**Out-of-Scope Uses**:
- ❌ High-stakes decisions without human review (e.g., automated account termination)
- ❌ Regulatory/compliance scenarios requiring interpretability guarantees
- ❌ Real-time latency-critical applications (batch inference recommended)
- ❌ Domains outside customer churn (model not transferable without retraining)

### Training Data
- **Source**: Generic churn dataset (ARFF format converted to CSV)
- **Size**: Varies by dataset; see logs for actual training sample size
- **Features**: Mix of numerical and categorical customer attributes
- **Target**: Binary churn indicator (encoded as 0=No, 1=Yes)
- **Preprocessing**: 
  - Median imputation for numerical features
  - Mode imputation + one-hot encoding for categorical features
  - Standard scaling for numerical features
- **Splits**: Train/validation/test with stratification (configurable via `config.json`)

### Performance Metrics
Metrics are computed on held-out validation and test sets:
- **Primary Metric**: ROC AUC (area under ROC curve)
- **Threshold Tuning**: Decision threshold optimized on validation set for:
  - F1 score maximization (default)
  - Precision-constrained recall
  - Top-k ratio targeting
- **Reported Metrics**: Accuracy, precision, recall, F1, PR AUC, confusion matrices
- **Artifacts**: See `models/model_diagnostics_{run_id}.txt` for detailed performance

**Typical Performance** (dataset-dependent):
- ROC AUC: ~0.75-0.85 (varies by data quality and feature richness)
- Note: Performance should be evaluated on your specific dataset

### Model Behavior & Limitations

**Known Limitations**:
1. **Data Quality Dependence**: Performance degrades with >70% missing values in key features
2. **Class Imbalance**: Default model may underpredict minority class; use `--class-weight balanced` if needed
3. **Feature Drift**: Model assumes feature distributions remain stable; monitor via `/drift` endpoint
4. **Temporal Patterns**: Random Forest doesn't capture time-series dependencies; add temporal features if relevant
5. **Interpretability**: Tree ensemble provides feature importances but not individual prediction explanations (consider SHAP for production)

**Bias & Fairness Considerations**:
- **Protected Attributes**: Model may learn from proxy variables correlated with sensitive attributes (age, gender, location)
- **Disparate Impact**: Evaluate fairness metrics across demographic groups before deployment
- **Feedback Loops**: Churn predictions may influence retention actions, creating self-fulfilling prophecies
- **Recommendation**: Conduct fairness audits and implement bias mitigation if deploying in regulated environments

**Edge Cases**:
- New customers with minimal history may receive unreliable predictions
- Feature values outside training distribution may produce unexpected outputs
- Schema mismatches are handled via alignment (adds NaN for missing features)

### Ethical Considerations
- **Consent**: Ensure customers consent to data usage for churn prediction
- **Transparency**: Consider disclosing to customers that retention offers are model-driven
- **Human Oversight**: Predictions should inform, not replace, human judgment in customer interactions
- **Privacy**: Remove PII from model inputs/outputs (use `--no-pii` flag); avoid logging sensitive data

### Maintenance & Monitoring
- **Retraining Cadence**: Retrain quarterly or when drift metrics exceed thresholds
- **Drift Detection**: Automated drift detection available via `POST /drift` endpoint
  - **Numerical features**: Monitors relative changes in mean and standard deviation
  - **Categorical features**: Uses Population Stability Index (PSI) to detect distribution shifts
  - **Prediction drift**: Tracks changes in positive class prediction rate
  - **Thresholds**: Configurable via environment variables (default: 10% for numerical/prediction, 0.1 PSI for categorical)
- **Monitoring**: Track prediction distribution, feature drift, and performance degradation via `/drift` and `/metrics` endpoints
- **Versioning**: All models are versioned with run IDs, git SHAs, and data hashes for reproducibility
- **Contact**: Mikkel Nielsen ([GitHub](https://github.com/Medesen/ML-portfolio))

### References
- Model Card methodology: Mitchell et al., "Model Cards for Model Reporting" (2019)
- Dataset: See `data/dataset.arff` source attribution

---

## Changelog

### v0.1.0 (October 2025)

**Major Features Added:**
- ✨ **Automated Drift Detection**: Comprehensive drift monitoring with PSI, statistical tests, and prediction distribution tracking
  - Reference statistics computed during training
  - `POST /drift` endpoint for real-time drift analysis
  - Configurable thresholds via environment variables
- ✨ **Model Card**: Complete model documentation following ML best practices
- ✨ **Performance Optimization**: Model caching in service (loaded once at startup)
- ✨ **Enhanced Testing**: Expanded test suite to 43+ tests covering all critical paths

**Improvements:**
- 🔧 Fixed deprecated scikit-learn parameter (`base_estimator` → `estimator`)
- 🔧 Proper dependency management: added `pydantic` and `pandera` to requirements
- 🔧 MLflow now correctly optional via `[mlflow]` extra
- 🔧 Schema alignment extracted to reusable function and applied to both CLI and service
- 🔧 Input validation added to service endpoints (max records, empty checks)
- 📝 Added MIT License
- 📝 Added repository URL to package metadata

**Documentation:**
- Enhanced README with feature highlights
- Added comprehensive Model Card
- Documented all environment variables
- Updated endpoint documentation with drift analysis examples

**Testing:**
- Added `test_schema_alignment.py` (9 tests)
- Added `test_service_validation.py` (8 tests)  
- Added `test_threshold_tuning.py` (6 tests)
- Added `test_drift_detection.py` (17 tests)

---

**Author**: Mikkel Nielsen  
**License**: MIT  
**Repository**: [github.com/Medesen/ML-portfolio](https://github.com/Medesen/ML-portfolio)

