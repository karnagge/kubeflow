import kfp
from kfp import dsl
from kfp.components import load_component_from_file

# Load pipeline components
data_gen_op = load_component_from_file("components/data_loading/component.yaml")
preprocessing_op = load_component_from_file("components/preprocessing/component.yaml")
training_op = load_component_from_file("components/training/component.yaml")


@dsl.pipeline(
    name="XGBoost Training Pipeline",
    description="End-to-end training pipeline for XGBoost model",
)
def xgboost_training_pipeline(
    pvc_name: str = "kubeflow-user-workspace", model_name: str = "xgboost-model"
):
    # Create volume operations
    vop = dsl.VolumeOp(
        name="create_pvc",
        resource_name="pipeline-pvc",
        size="5Gi",
        modes=dsl.VOLUME_MODE_RWO,
    )

    # Data generation step
    data_gen_task = data_gen_op().add_pvolumes({"/data": vop.volume})

    # Preprocessing step
    preprocess_task = (
        preprocessing_op(
            train_data=data_gen_task.outputs["train_data"],
            val_data=data_gen_task.outputs["val_data"],
            test_data=data_gen_task.outputs["test_data"],
        )
        .add_pvolumes({"/data": vop.volume})
        .after(data_gen_task)
    )

    # Get best hyperparameters from Katib experiment
    hyper_params = {
        "objective": "binary:logistic",
        "eval_metric": ["error", "auc", "logloss"],
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 500,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    # Training step
    train_task = (
        training_op(
            train_data=preprocess_task.outputs["processed_train_data"],
            val_data=preprocess_task.outputs["processed_val_data"],
            hyperparameters=hyper_params,
        )
        .add_pvolumes(
            {"/data": vop.volume, "/models": dsl.PipelineVolume(pvc=pvc_name)}
        )
        .after(preprocess_task)
    )

    # Deploy model using KServe
    deploy_model = dsl.ResourceOp(
        name="deploy-model",
        k8s_resource={
            "apiVersion": "serving.kubeflow.org/v1beta1",
            "kind": "InferenceService",
            "metadata": {"name": model_name, "namespace": "kubeflow-user"},
            "spec": {
                "predictor": {
                    "xgboost": {
                        "storageUri": f"pvc://{pvc_name}/models",
                        "resources": {
                            "requests": {"cpu": "2", "memory": "4Gi"},
                            "limits": {"cpu": "4", "memory": "8Gi"},
                        },
                    }
                }
            },
        },
    ).after(train_task)


# Compile the pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=xgboost_training_pipeline,
        package_path="xgboost_training_pipeline.yaml",
    )
