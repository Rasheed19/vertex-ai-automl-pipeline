from kfp.dsl import Artifact, ClassificationMetrics, Input, Metrics, Output, component
from typing_extensions import NamedTuple


@component(
    base_image="gcr.io/deeplearning-platform-release/tf2-cpu.2-6:latest",
    packages_to_install=["google-cloud-aiplatform"],
)
def classification_model_eval_metrics(
    project: str,
    location: str,
    thresholds_dict_str: str,
    model: Input[Artifact],
    metrics: Output[Metrics],
    metricsc: Output[ClassificationMetrics],
) -> NamedTuple("Outputs", [("dep_decision", str)]):  # Return parameter.
    # Note all functions and packages used inside
    # a component must be defined and impoted
    # inside the component (at least for the version of google-ai-platform used
    # in this project)

    import json
    import logging
    from google.cloud import aiplatform

    aiplatform.init(project=project)

    # Fetch model eval info
    def get_eval_info(model):
        response = model.list_model_evaluations()
        metrics_list = []
        metrics_string_list = []
        for evaluation in response:
            evaluation = evaluation.to_dict()
            print("model_evaluation")
            print(" name:", evaluation["name"])
            print(" metrics_schema_uri:", evaluation["metricsSchemaUri"])
            metrics = evaluation["metrics"]
            for metric in metrics.keys():
                logging.info("metric: %s, value: %s", metric, metrics[metric])
            metrics_str = json.dumps(metrics)
            metrics_list.append(metrics)
            metrics_string_list.append(metrics_str)

        return (
            evaluation["name"],
            metrics_list,
            metrics_string_list,
        )

    # Use the given metrics threshold(s) to determine whether the model is
    # accurate enough to deploy.
    def classification_thresholds_check(metrics_dict, thresholds_dict):
        for k, v in thresholds_dict.items():
            logging.info("k {}, v {}".format(k, v))
            if k in ["auRoc", "auPrc"] and metrics_dict[k] < v:
                logging.info("{} < {}; returning False".format(metrics_dict[k], v))
                return False
        logging.info("threshold checks passed.")
        return True

    def log_metrics(metrics_list, metricsc):
        test_confusion_matrix = metrics_list[0]["confusionMatrix"]
        logging.info("rows: %s", test_confusion_matrix["rows"])

        # log the ROC curve
        fpr = []
        tpr = []
        thresholds = []
        for item in metrics_list[0]["confidenceMetrics"]:
            fpr.append(item.get("falsePositiveRate", 0.0))
            tpr.append(item.get("recall", 0.0))
            thresholds.append(item.get("confidenceThreshold", 0.0))
        print(f"fpr: {fpr}")
        print(f"tpr: {tpr}")
        print(f"thresholds: {thresholds}")
        metricsc.log_roc_curve(fpr, tpr, thresholds)

        # log the confusion matrix
        annotations = []
        for item in test_confusion_matrix["annotationSpecs"]:
            annotations.append(item["displayName"])
        logging.info("confusion matrix annotations: %s", annotations)
        metricsc.log_confusion_matrix(
            annotations,
            test_confusion_matrix["rows"],
        )

        # log textual metrics info as well
        for metric in metrics_list[0].keys():
            if metric != "confidenceMetrics":
                val_string = json.dumps(metrics_list[0][metric])
                metrics.log_metric(metric, val_string)

    logging.getLogger().setLevel(logging.INFO)

    # extract the model resource name from the input Model Artifact
    model_resource_path = model.metadata["resourceName"]
    logging.info("model path: %s", model_resource_path)

    # Get the trained model resource
    model = aiplatform.Model(model_resource_path)

    # Get model evaluation metrics from the the trained model
    eval_name, metrics_list, metrics_str_list = get_eval_info(model)
    logging.info("got evaluation name: %s", eval_name)
    logging.info("got metrics list: %s", metrics_list)
    log_metrics(metrics_list, metricsc)

    thresholds_dict = json.loads(thresholds_dict_str)
    deploy = classification_thresholds_check(metrics_list[0], thresholds_dict)
    dep_decision = "true" if deploy else "false"
    logging.info("deployment decision is %s", dep_decision)

    return (dep_decision,)
