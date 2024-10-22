FROM python:3.11
WORKDIR /Users/andylenguyen/Documents/MLOps_House_Price_Predict/
COPY . . 
# Python dependency management
RUN pip install pipenv
# Install dependencies in Pipfile.lock into a virtual env on image
RUN pipenv install
EXPOSE 5000
CMD [ "pipenv", "run", "python", "model_api.py" ]




