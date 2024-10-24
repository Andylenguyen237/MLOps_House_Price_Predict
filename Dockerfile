FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the Pipfile and Pipfile.lock to install dependencies
COPY Pipfile Pipfile.lock /app/

# Install pipenv to manage dependencies
RUN pip install pipenv

# Install dependencies from Pipfile.lock
RUN pipenv install 

# Copy the entire app, including model_api.py and the dataset
COPY . /app

# Expose the port 
EXPOSE 5000

# Run the Flask application
CMD ["pipenv", "run", "python", "model_api.py"]





