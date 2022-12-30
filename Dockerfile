FROM python:3.8

WORKDIR /
ADD . /

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
ENV PORT 8000

# Run the application:
CMD ["gunicorn", "app:app", "--config=config.py"]