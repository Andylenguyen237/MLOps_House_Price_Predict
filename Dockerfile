FROM python:3.11

# Working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install pipenv
RUN pip install pipenv

# Install the dependencies from Pipfile.lock
RUN pipenv install 

# Port on which the Flask app will run
EXPOSE 5000

# Command to run the application
CMD [ "pipenv", "run", "python", "model_api.py" ]





