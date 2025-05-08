# Project Title

A brief description of this project.

## Prerequisites

- Docker

## Getting Started

### Building the Docker Image

To build the Docker image, navigate to the project's root directory (where the `Dockerfile` is located) and run the following command:

```bash
docker build -t your-image-name .
```

### Running the Docker Container

Once the image is built, you can run a container using:

```bash
docker run -p 8000:8000 -p 8888:8888 your-image-name
```

This will start:

- JupyterLab on port `8888`
- The vLLM REST API on port `8000`

## Environment Variables

The following environment variables can be set when running the Docker container (e.g., using the `-e` flag with `docker run`):

- `MODEL_NAME`: The path or Hugging Face model ID for the model to be served by vLLM. Defaults to `/app/models/default-model`.
- `WANDB_API_KEY`: Your Weights & Biases API key for experiment tracking.
- `WANDB_MODE`: Set to `online` or `offline` for Weights & Biases. Defaults to `online`.
- `HUGGINGFACE_TOKEN`: Your Hugging Face Hub token, if needed for downloading private models.

## GPU Monitoring

A cron job is set up to monitor GPU usage. The script `check_gpu_usage.sh` is responsible for this. (Further details on what this script does and where it logs can be added here).

## Accessing Services

- **JupyterLab**: Open your browser and navigate to `http://localhost:8888`
- **vLLM API**: The API will be accessible at `http://localhost:8000`. You can find more about the vLLM API endpoints [here](https://vllm.readthedocs.io/en/latest/serving/openai_compatible_server.html).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
