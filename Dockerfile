FROM continuumio/miniconda:latest

WORKDIR /app

# Copy script, model and input
COPY main.py .
COPY model.pkl .
COPY input ./input

# Handle conda environment dependencies
COPY environment.yml .
RUN conda env create -f environment.yml

# On default conda will not run, override conda to be used with /bin/bash instead of /bin/sh
SHELL ["conda", "run", "-n", "image_labeler_env", "/bin/bash", "-c"]

# Run the application
ENTRYPOINT ["conda", "run", "-n", "image_labeler_env", "python", "main.py"]