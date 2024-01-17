import kfp
from kfp import dsl
from dotenv import dotenv_values
from components import classification_model_eval_metrics


config = dotenv_values(".env")


# set path for storing the pipeline artifacts
PIPELINE_NAME = "automl-tabular-beans-training"
PIPELINE_ROOT = f"{config['BUCKET_URI']}/pipeline_root/beans"


@kfp.dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(
    bq_source: str,
    DATASET_DISPLAY_NAME: str,
    TRAINING_DISPLAY_NAME: str,
    MODEL_DISPLAY_NAME: str,
    ENDPOINT_DISPLAY_NAME: str,
    MACHINE_TYPE: str,
    project: str,
    gcp_region: str,
    thresholds_dict_str: str,
):
    # Import these libraries which are used inside this
    # pipeline here, don't put them outside the pipeline.

    # There is an exception for the component which has been
    # configured in a different module.

    from google_cloud_pipeline_components.v1.automl.training_job import \
        AutoMLTabularTrainingJobRunOp
    from google_cloud_pipeline_components.v1.dataset.create_tabular_dataset.component import \
        tabular_dataset_create as TabularDatasetCreateOp
    from google_cloud_pipeline_components.v1.endpoint.create_endpoint.component import \
        endpoint_create as EndpointCreateOp
    from google_cloud_pipeline_components.v1.endpoint.deploy_model.component import \
        model_deploy as ModelDeployOp
    
    # create a Vertex AI dataset from a data stored in BigQuery
    dataset_create_op = TabularDatasetCreateOp(
        project=project,
        location=gcp_region,
        display_name=DATASET_DISPLAY_NAME,
        bq_source=bq_source,
    )
    
    # create AutoML training job
    training_op = AutoMLTabularTrainingJobRunOp(
        project=project,
        location=gcp_region,
        display_name=TRAINING_DISPLAY_NAME,
        optimization_prediction_type="classification",
        optimization_objective="minimize-log-loss",
        budget_milli_node_hours=1000,
        model_display_name=MODEL_DISPLAY_NAME,
        column_specs={
            "Area": "numeric",
            "Perimeter": "numeric",
            "MajorAxisLength": "numeric",
            "MinorAxisLength": "numeric",
            "AspectRation": "numeric",
            "Eccentricity": "numeric",
            "ConvexArea": "numeric",
            "EquivDiameter": "numeric",
            "Extent": "numeric",
            "Solidity": "numeric",
            "roundness": "numeric",
            "Compactness": "numeric",
            "ShapeFactor1": "numeric",
            "ShapeFactor2": "numeric",
            "ShapeFactor3": "numeric",
            "ShapeFactor4": "numeric",
            "Class": "categorical",
        },
        dataset=dataset_create_op.outputs["dataset"],
        target_column="Class",
    )
    
    # evaluate model using the evaluation component
    model_eval_task = classification_model_eval_metrics(
        project=project,
        location=gcp_region,
        thresholds_dict_str=thresholds_dict_str,
        model=training_op.outputs["model"],
    )

    with dsl.Condition(
        model_eval_task.outputs["dep_decision"] == "true",
        name="deploy_decision",
    ):
        
        # create an endpoint
        endpoint_op = EndpointCreateOp(
            project=project,
            location=gcp_region,
            display_name=ENDPOINT_DISPLAY_NAME,
        )
        
        # deploy endpoint if it satifies accuracy threshold
        ModelDeployOp(
            model=training_op.outputs["model"],
            endpoint=endpoint_op.outputs["endpoint"],
            dedicated_resources_min_replica_count=1,
            dedicated_resources_max_replica_count=1,
            dedicated_resources_machine_type=MACHINE_TYPE,
        )
     