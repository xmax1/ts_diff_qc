FROM python:3.10 as build


# staging: Installs poetry and copy relevant source files
# development: Install our project in editable mode
# build: Build our project into a wheel file
# production: Clean Python image that installs our built wheel

### builder image (means build environment is the same everytime)

ARG APP_NAME=test-app
ARG APP_PATH=/opt/$APP_NAME
ARG PYTHON_VERSION=3.10.10

RUN mkdir "$APP_PATH"
COPY . "$APP_PATH"

RUN apt-get update && \
    apt-get install -y \
    vim \
    less

ENV \
    PYTHONDONTWRITEBYTECODE=1 \ 
    PYTHONUNBUFFERED=1 \ 
    PYTHONFAULTHANDLER=1 
ENV \
    POETRY_VIRTUALENVS_IN_PROJECT=true \ 
    POETRY_NO_INTERACTION=1 \
    PATH=/opt/poetry/bin:"$PATH"

RUN apt-get install -y curl

# Prevents Python from writing pyc files to disc
# Prevents Python from buffering stdout and stderr
# Enables the faulthandler module which dumps the Python traceback
# Install poetry in /opt/poetry
# Create a virtualenv in the project directory
# Prevents poetry from asking any interactive question
RUN echo "Installing poetry"

WORKDIR "$APP_PATH"
RUN echo "$CWD"
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME="/opt/poetry" python3 -

RUN pwd && ls -la 

RUN echo "$PATH"
RUN poetry --version
RUN python --version

COPY ./poetry.lock ./poetry.lock
COPY ./pyproject.toml ./pyproject.toml
COPY ./$APP_NAME ./$APP_NAME

# ---------------------------- #

FROM build as stage

RUN poetry build --format wheel
RUN poetry export --format env.txt --output constraints.txt --without-hashes

FROM stage as production

ARG APP_NAME
ARG APP_PATH

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Disable pip cache
# Disable pip version check
ENV \
    PIP_NO_CACHE_DIR=off \ 
    PIP_DISABLE_PIP_VERSION_CHECK=on \ 
    PIP_DEFAULT_TIMEOUT=100 

WORKDIR $APP_PATH
COPY --from=build $APP_PATH/dist/*.whl ./
COPY --from=build $APP_PATH/constraints.txt ./
RUN pip install ./$APP_NAME*.whl --constraint constraints.txt