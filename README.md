# vertex-ai-automl-pipeline
Vertex AI e2e pipeline with classification problem case using AutoML.

This project is based on the Google's demo which can be 
found in https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/automl_tabular_classification_beans.ipynb

## Folder analysis
1. `config` contains the component and pipeline configuration files
1. `components` contains veterx component python files
1. `pipelines` contains veterx pipeline python files
1. `utils` contains helper functions 

## Set up for running locally
1. Clone the repository by running
    ```
    git clone https://github.com/Rasheed19/vertex-ai-automl-pipeline
    ```
1. Navigate to the root folder, i.e., `vertex-ai-automl-pipeline` and create a python virtual environment by running
    ```
    python3.10 -m venv .venv
    ``` 
1. Activate the virtual environment by running
    ```
    source .venv/bin/activate
    ```
1. Upgrade `pip` by running 
   ```
   pip install --upgrade pip
   ``` 
1. Install all the required Python libraries by running 
    ```
    pip install -r requirements.txt
    ```
1. Download the beans data from hhttps://archive.ics.uci.edu/dataset/602/dry+bean+dataset. Convert it to csv and upload it to the BigQuery

1. Create a file named `.env` in the root folder and store the following variables related to your GCP:
    ```
    PROJECT_ID=your-project-id
    REGION=your-project-region
    BUCKET_URI=gs://your-project-name
    SERVICE_ACCOUNT=your-service-account
    ```
1. Run the following commands in your terminal to configure the pipeline run on the Vertex AI (make sure 
   `gcloud CLI` is installed on your computer):
   1. Login:
       ```
       gcloud auth login
       ```
   1.  Configure the login to use your prefered project:
        ```
        gcloud config set project your-prpject-id
        ```
    1. Get and save your user account credentials:
          ```
          gcloud auth application-default login
          ```
    1. Grant access to the pipeline to use your storage bucket
        ```
        gsutil iam ch serviceAccount:your-service-account:roles/storage.objectCreator gs://your-project-name
        ```

        ```
        gsutil iam ch user:your-gmail-address:objectCreator gs://your-project-name
        ```

1. Then run the pipeline that trains, registers, and deploys a trained model to the Vertex AI endpoint
   by running one of the following customised commands in your terminal:
    1. Run the pipeline with default options
        ```
        python run.py
        ```
            
    1. Run the pipeline with Quality Gate for test AU ROC set at 95% for test set. If the threshold fails, then the model won't be deployed to an endpoint.
       ```
       python run.py --min-test-accuracy 0.95
       ````