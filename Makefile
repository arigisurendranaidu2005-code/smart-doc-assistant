.PHONY: install run-api run-frontend run-all test docker-build docker-up lint clean

# Variables
PYTHON = python
UVICORN = uvicorn
STREAMLIT = streamlit
PORT_API = 8000
PORT_UI = 8501

install:
	pip install -r requirements.txt

run-api:
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port $(PORT_API)

run-frontend:
	$(STREAMLIT) run app/main.py --server.port $(PORT_UI)

run-all:
	start /B $(UVICORN) api.main:app --reload --host 0.0.0.0 --port $(PORT_API) & $(STREAMLIT) run app/main.py --server.port $(PORT_UI)

test:
	pytest tests/ -v

docker-build:
	docker build -t smart-doc-assistant .

docker-up:
	docker-compose up --build -d

lint:
	flake8 api app tests
	black --check api app tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf data/chroma_db
