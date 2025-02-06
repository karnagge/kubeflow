# End-to-End Machine Learning with Kubeflow

This project demonstrates an end-to-end machine learning workflow using Kubeflow components. It includes data generation, preprocessing, model training with XGBoost, hyperparameter tuning with Katib, and model serving with KServe.

## Project Structure

```
e2e-kubeflow/
├── notebooks/                      # Jupyter notebooks for development and testing
│   ├── 01-data-generation.ipynb   # Generate synthetic data
│   ├── 02-data-preprocessing.ipynb # Data preprocessing steps
│   └── 03-model-development.ipynb # Model development and testing
├── experiments/                    # Katib experiment configurations
│   ├── katib-experiment.yaml      # Hyperparameter tuning experiment
│   └── hyperparameter-tuning/     # Training scripts for Katib
├── pipelines/                      # Kubeflow pipeline definitions
│   ├── components/                # Pipeline components
│   │   ├── data_loading/         # Data generation component
│   │   ├── preprocessing/        # Data preprocessing component
│   │   ├── training/            # Model training component
│   │   └── evaluation/          # Model evaluation component
│   └── training-pipeline.py      # Main pipeline definition
└── models/                        # Model artifacts
    └── serving/                  # KServe deployment configurations
```

## Prerequisites

1. Kubeflow 1.7+ installed and configured
2. Access to a Kubernetes cluster
3. Python 3.8+
4. Required Python packages:
   - kfp (Kubeflow Pipelines SDK)
   - xgboost
   - scikit-learn
   - pandas
   - numpy

## Setup Instructions

1. Clone this repository:
```bash
git clone <repository-url>
cd e2e-kubeflow
```

2. Create a Python virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Development Notebooks

The notebooks in the `notebooks/` directory can be used to:
- Generate and explore synthetic data
- Develop and test preprocessing steps
- Experiment with the XGBoost model

### 2. Hyperparameter Tuning with Katib

1. Apply the Katib experiment:
```bash
kubectl apply -f experiments/katib-experiment.yaml
```

2. Monitor the experiment:
```bash
kubectl get experiment xgboost-hpo -n kubeflow-user
```

### 3. Running the Pipeline

1. Compile the pipeline:
```bash
python pipelines/training-pipeline.py
```

2. Upload the generated `xgboost_training_pipeline.yaml` to the Kubeflow Pipelines UI

3. Create a new run with the following parameters:
   - PVC name: Your persistent volume claim name
   - Model name: Name for your deployed model

### 4. Model Serving

The model will be automatically deployed using KServe after successful pipeline execution. You can verify the deployment:

```bash
kubectl get inferenceservice xgboost-model -n kubeflow-user
```

## Model Details

- Model: XGBoost Classifier
- Input: 20 numerical features
- Output: Binary classification (0 or 1)
- Target Performance: >90% accuracy
- Metrics Tracked:
  - Accuracy
  - AUC-ROC
  - Log Loss

## Pipeline Components

1. **Data Generation**
   - Creates synthetic tabular data
   - Splits into train/validation/test sets
   - Introduces controlled noise and missing values

2. **Preprocessing**
   - Handles missing values using median imputation
   - Scales features using StandardScaler
   - Creates interaction features

3. **Training**
   - Uses best hyperparameters from Katib
   - Implements early stopping
   - Saves model artifacts and metrics

4. **Serving**
   - Deploys model using KServe
   - Includes preprocessing transformers
   - Auto-scales based on load

## Monitoring and Maintenance

- Monitor model performance through Kubeflow's metrics UI
- Check KServe logs for inference issues:
```bash
kubectl logs -n kubeflow-user -l serving.kubeflow.org/inferenceservice=xgboost-model
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.