FROM python:3.9

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app
RUN chmod -R a+rwx /app

# Run your application
CMD ["python", "test.py"]