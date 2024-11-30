FROM python:3.10-slim
WORKDIR /home/energy_analysis_toolbox
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    build-essential bash-completion git
RUN python -m pip install uv
RUN uv venv
COPY . /home/energy_analysis_toolbox/
RUN uv pip install -r /home/energy_analysis_toolbox/pyproject.toml
RUN git config --global --add safe.directory /home/energy_analysis_toolbox
ENV SHELL "/bin/bash"
CMD ["/bin/bash"]
