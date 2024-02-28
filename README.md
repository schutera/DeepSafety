# Deep Safety
This is a repository to study safety concepts for deep learning in safety-critical applications. The repository is part of the lecture at DHBW Ravensburg. 

This course includes a project whose goal is to validate a machine learning based system for traffic sign classification that is assumed to be part of a self-driving vehicle. During the course, you will you work on several assignments that cover specific aspects of safety for deep learning and prepare you for the final project.

## Quickstart
1. Clone the repository:
    ```bash
    git clone git@github.com:schutera/DeepSafety.git
    ```
2. Create a new virtual environment and install dependencies.

    Either using Conda (recommended):
    ```bash
    conda env create --file environment.yaml
    ```
    or using venv:
    ```bash
    python -m venv deepsafety_env
    source deepsafety_env/bin/activate
    pip install -r requirements.txt
    ```
    Please refer to the [PyTorch Get Started guide](https://pytorch.org/get-started/locally/) for installation details if you want to use a GPU.
3. Activate your virtual environment, e.g., Conda:
    ```bash
    conda activate deepsafety_env
    ```
4. Train a new model using the default parameters:
    ```bash
    python -m train
    ```
    If you want to use custom parameters, you can use the `-h` argument to get an overview over the possible arguments.
5. Launch the MLFlow UI to view your logged run in the tracking UI. From a terminal in the repository root directory run:
    ```bash
    mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
    ```
    Then, navigate to http://localhost:8080 in your browser to view the results.
6. Evaluate your trained model:
    ```bash
    python -m eval --data-dir <path_to_evaluation_data> --run-id <MLFlow_run_id>
    ```