from google.cloud import aiplatform, bigquery
import click
from components import classification_model_eval_metrics, component_compiler
from pipelines import pipeline, pipeline_compiler
from utils import generate_uuid
from dotenv import dotenv_values


@click.command(
    help="""
    Vertex AI e2e pipeline with classification problem case.

    This project is based on the Google's demo which can be 
    found in https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/automl_tabular_classification_beans.ipynb


    Run the odel training pipeline with various
    options.

    Examples:

    \b
    # Run the pipeline with default options
    python run.py
                
    \b
    # Run the pipeline with Quality Gate for test AU ROC set at 95% for test set.
    # If the threshold fails, then the model won't be deployed to an endpoint.
    python run.py --min-test-accuracy 0.95

    """
)
@click.option(
    "--min-test-accuracy",
    default=0.95,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum test accuracy to pass to the model evaluator.",
)
def main(
    min_test_accuracy: float = 0.95,
):
    config = dotenv_values(".env")

    # generate UUID
    UUID = generate_uuid()

    # define some constants for default display-names
    PIPELINE_ROOT = f"{config['BUCKET_URI']}/pipeline_root/beans"
    BQ_SOURCE = f"{config['PROJECT_ID']}.dry_bean_dataset.beans"  # make sure this matches with your uploaded dataset in BigQuery
    PIPELINE_DISPLAY_NAME = f"pipeline_beans_{UUID}"
    DATASET_DISPLAY_NAME = f"dataset_beans_{UUID}"
    MODEL_DISPLAY_NAME = f"model_beans_{UUID}"
    TRAINING_DISPLAY_NAME = f"automl_training_beans_{UUID}"
    ENDPOINT_DISPLAY_NAME = f"endpoint_beans_{UUID}"

    # set machine type
    MACHINE_TYPE = "n1-standard-4"

    # intialize project
    aiplatform.init(
        project=config["PROJECT_ID"],
        location=config["REGION"],
        staging_bucket=config["BUCKET_URI"],
    )

    # compile component
    component_compiler(
        component=classification_model_eval_metrics,
        config_path_name="./config/tabular_eval_component.yaml",
    )

    # compile pipeline
    pipeline_compiler(
        pipeline=pipeline,
        config_path_name="./config/tabular_classification_pipeline.yaml",
    )

    # intialize training job
    client = bigquery.Client()
    bq_region = client.get_table(BQ_SOURCE).location.lower()
    try:
        assert bq_region in config["REGION"]
        print(f"Region validated: {config['REGION']}")
    except AssertionError:
        print(
            "Please make sure the region of BigQuery (source) and that of the pipeline are the same."
        )

    # configure the pipeline
    job = aiplatform.PipelineJob(
        display_name=PIPELINE_DISPLAY_NAME,
        template_path="./config/tabular_classification_pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project": config["PROJECT_ID"],
            "gcp_region": config["REGION"],
            "bq_source": f"bq://{BQ_SOURCE}",
            "thresholds_dict_str": '{{"auRoc": {}}}'.format(
                min_test_accuracy
            ),  # note the double quotes around auRoc, should comply with json rule
            "DATASET_DISPLAY_NAME": DATASET_DISPLAY_NAME,
            "TRAINING_DISPLAY_NAME": TRAINING_DISPLAY_NAME,
            "MODEL_DISPLAY_NAME": MODEL_DISPLAY_NAME,
            "ENDPOINT_DISPLAY_NAME": ENDPOINT_DISPLAY_NAME,
            "MACHINE_TYPE": MACHINE_TYPE,
        },
        enable_caching=False,
    )

    # submit job for training on vertex ai
    job.submit()


if __name__ == "__main__":
    main()
