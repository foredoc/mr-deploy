# mr-deploy
Deploy MedGemma-Radiology on GCP Cloud Run

## Deploying Your MedGemma Streamlit App to Google Cloud Run with GPU

These instructions will guide you through deploying your Streamlit application, which uses MedGemma and Gemini TTS, to Google Cloud Run with an NVIDIA T4 GPU.

**Prerequisites:**

1.  **Google Cloud SDK (gcloud CLI):** [Install and initialize](https://cloud.google.com/sdk/docs/install).
2.  **Docker:** [Install Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine. Ensure it's running.
3.  **GCP Project:** A Google Cloud Project with billing enabled. Note your Project ID.
4.  **API Keys:**
    * `HF_TOKEN`: Your Hugging Face access token (if `google/medgemma-4b-it` is private/gated or to avoid rate limits).
    * `GOOGLE_API_KEY`: Your API key for Google Gemini services.
5.  **Files:** Ensure you have `app.py`, `requirements.txt`, `Dockerfile`, `start.sh`, and optionally `.env.placeholder` in the same directory.

---

**Step 1: Set Up Your GCP Environment**

1.  **Authenticate gcloud:**
    ```bash
    gcloud auth login
    gcloud auth application-default login
    ```

2.  **Set your Project ID:**
    ```bash
    export PROJECT_ID="your-gcp-project-id" # Replace with your actual project ID
    gcloud config set project $PROJECT_ID
    ```

3.  **Enable Necessary APIs:**
    ```bash
    gcloud services enable run.googleapis.com \
        artifactregistry.googleapis.com \
        iam.googleapis.com \
        secretmanager.googleapis.com \
        compute.googleapis.com # For GPU availability checks, indirectly used by Cloud Run
    ```

---

**Step 2: Store API Keys as Secrets in Secret Manager**

It's crucial to store sensitive API keys securely.

1.  **Create Secret for Hugging Face Token:**
    ```bash
    echo -n "your_hugging_face_token_here" | \
    gcloud secrets create HF_TOKEN_SECRET --replication-policy="automatic" --data-file=-
    ```
    Replace `"your_hugging_face_token_here"` with your actual token.

2.  **Create Secret for Google API Key:**
    ```bash
    echo -n "your_google_gemini_api_key_here" | \
    gcloud secrets create GOOGLE_API_KEY_SECRET --replication-policy="automatic" --data-file=-
    ```
    Replace `"your_google_gemini_api_key_here"` with your actual key.

3.  **(Optional) Create Secret for Gemini TTS Model Name (if configurable):**
    If you made `GEMINI_TTS_MODEL_NAME` an environment variable in your app:
    ```bash
    # echo -n "gemini-2.5-flash-preview-tts" | \
    # gcloud secrets create GEMINI_TTS_MODEL_SECRET --replication-policy="automatic" --data-file=-
    ```

4.  **Grant Cloud Run Service Account Access to Secrets:**
    When you deploy your Cloud Run service, it will use a service account (usually the default Compute Engine service account or a custom one). This account needs permission to access the secrets.
    Get your project number:
    ```bash
    export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
    ```
    The default Compute Engine service account is `PROJECT_NUMBER-compute@developer.gserviceaccount.com`.
    Grant access:
    ```bash
    gcloud secrets add-iam-policy-binding HF_TOKEN_SECRET \
        --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
        --role="roles/secretmanager.secretAccessor"

    gcloud secrets add-iam-policy-binding GOOGLE_API_KEY_SECRET \
        --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
        --role="roles/secretmanager.secretAccessor"

    # If you created GEMINI_TTS_MODEL_SECRET:
    # gcloud secrets add-iam-policy-binding GEMINI_TTS_MODEL_SECRET \
    # --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    # --role="roles/secretmanager.secretAccessor"
    ```
    *Note: If you use a custom service account for Cloud Run, replace the `--member` value accordingly.*

---

**Step 3: Build and Push Docker Image to Artifact Registry**

1.  **Create an Artifact Registry Docker Repository:**
    Choose a region that supports T4 GPUs for Cloud Run (e.g., `us-central1`, `europe-west4`).
    ```bash
    export REGION="us-central1" # Example region
    export REPOSITORY_NAME="medgemma-repo" # Choose a name for your repository

    gcloud artifacts repositories create $REPOSITORY_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="Docker repository for MedGemma app"
    ```

2.  **Configure Docker to Authenticate with Artifact Registry:**
    ```bash
    gcloud auth configure-docker ${REGION}-docker.pkg.dev
    ```

3.  **Build the Docker Image:**
    Navigate to the directory containing your `Dockerfile` and other application files.
    ```bash
    export IMAGE_NAME="medgemma-app"
    export IMAGE_TAG="latest"
    export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

    docker build -t $IMAGE_URI .
    ```
    This step can take a while, especially when `bitsandbytes` or PyTorch are being installed/compiled.
    *Troubleshooting Note:* If `torch==2.7.0` or `transformers==4.52.3` (or their CUDA variants) are not found by pip, the build will fail. You might need to use different, available versions or find specific installation instructions for those pre-release/very new versions.

4.  **Push the Docker Image to Artifact Registry:**
    ```bash
    docker push $IMAGE_URI
    ```

---

**Step 4: Deploy to Cloud Run with GPU**

1.  **Deploy the Service:**
    * Replace `your-service-name` with a name for your Cloud Run service.
    * Ensure the `--region` matches where your Artifact Registry and GPU availability are.
    * `--cpu-boost` and increased `--cpu`, `--memory` can help with model loading times and performance. T4 typically has 16GB GPU RAM.
    * `--concurrency` set to 1 is often safer for GPU models unless you've specifically designed for concurrent requests on a single instance.
    * `--timeout` should be sufficient for model loading and inference.
    * `--set-secrets` maps the Secret Manager secrets to environment variables in your container.

    **Explanation of Flags:**
    * `--image`: Your image in Artifact Registry.
    * `--platform=managed`: Use managed Cloud Run.
    * `--region`: GCP region.
    * `--allow-unauthenticated`: Makes the service publicly accessible. Remove if you want IAM-based auth.
    * `--port=8501`: The port your Streamlit app listens on inside the container.
    * `--cpu=4 --memory=16Gi`: Adjust based on needs. Model loading can be resource-intensive.
    * `--min-instances=0`: Allows scaling to zero to save costs (cold starts will be slower). Set to `1` for faster responses if budget allows.
    * `--max-instances=2`: Controls maximum scaling.
    * `--concurrency=1`: Recommended for many ML models to avoid overwhelming a single instance.
    * `--timeout=900s`: (15 minutes) Max request timeout. Adjust as needed for long inferences or model loads.
    * `--execution-environment=gen2`: Second-generation execution environment is generally recommended.
    * `--update-secrets`: Maps secrets to environment variables. The format is `ENV_VAR_NAME=SECRET_NAME:VERSION`.
    * `--gpu=type=nvidia-l4`: **This is how you request a GPU.** It's passed as an argument to the underlying infrastructure.

   ```bash
    gcloud run deploy $SERVICE_NAME \
        --image=$IMAGE_URI \
        --platform=managed \
        --region=$REGION \
        --allow-unauthenticated \
        --port=8501 \
        --cpu=4 \
        --memory=24Gi \
        --min-instances=0 \
        --max-instances=2 \
        --concurrency=1 \
        --timeout=900s \
        --execution-environment=gen2 \
        --update-secrets=HF_TOKEN=HF_TOKEN_SECRET:latest,GOOGLE_API_KEY=GOOGLE_API_KEY_SECRET:latest \
        --gpu-type=nvidia-l4 \
        --service-account="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
        # If you added GEMINI_TTS_MODEL_SECRET:
        # ,GEMINI_TTS_MODEL_NAME=GEMINI_TTS_MODEL_SECRET:latest (append to --update-secrets value)
```

2.  **Monitor Deployment:**
    The deployment process will output a URL for your service once it's ready. You can also monitor it in the Google Cloud Console under Cloud Run. The first deployment with GPU and model download might take several minutes.

---

**Step 5: Access Your Application**

Once deployed, `gcloud` will provide a service URL. Open this URL in your browser.
You can also find the URL in the Cloud Run section of the GCP Console.

---

**Troubleshooting & Important Notes:**

* **CUDA & PyTorch/bitsandbytes Compatibility:** This is the most common source of issues. Ensure your `Dockerfile`'s base CUDA version, the PyTorch installation command (with CUDA suffix like `cu121`), and the `bitsandbytes` version are all compatible. The versions specified in `requirements.txt` for `torch` and `transformers` are very new; if they cause build or runtime issues, try decrementing to the latest stable versions known to work with CUDA 12.1 and `bitsandbytes`.
* **Model Download:** The MedGemma model will be downloaded the first time an instance starts. This can take time and requires network access. Subsequent requests to the same instance (if warm) will be faster.
* **GPU Availability:** T4 GPUs are not available in all regions. Ensure your chosen region supports them for Cloud Run.
* **Quotas:** You might need to request GPU quota increases for your project in the desired region.
* **Cold Starts:** If `min-instances` is 0, the first request after a period of inactivity will trigger a "cold start," which includes instance provisioning, image download, and model loading. This can be slow. Set `min-instances=1` for faster, more consistent response times at a higher cost.
* **Logs:** Check Cloud Run logs in the GCP Console for troubleshooting. `gcloud run services logs tail $SERVICE_NAME --region $REGION --project $PROJECT_ID`
* **`app.py` adjustments for MedGemma `pipe`:**
    * The task for `pipeline` in `app.py` was `image-text-to-text`. Common tasks are `image-to-text` or `visual-question-answering`. Ensure `google/medgemma-4b-it` supports the specified task name and input message format. I've slightly adjusted `app.py` comments around this and the TTS part, but the core logic you provided is kept.
    * The parsing of the output from `pipe` might need adjustment based on the actual structure returned by your specific MedGemma pipeline configuration.
* **Gemini TTS Model Name:** The model name `gemini-2.5-flash-preview-tts` and voice `Kore` in `app.py` are very specific. Double-check these are correct and available for your `GOOGLE_API_KEY` and `google-generativeai` library version. I've added comments in `app.py` regarding this. If issues arise, you might need to use standard model names/voices documented for the Gemini API.
* **Resource Allocation:** 4 vCPU and 16Gi RAM with a T4 GPU is a starting point. Monitor your application's performance and adjust as needed. The T4 GPU itself has 16GB of VRAM.

This comprehensive guide should help you get your application running on Cloud Run with GPU support. Remember to replace placeholder values with your actual project details and API keys.

