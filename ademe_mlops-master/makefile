install:
    pip install --upgrade pip && pip install -r requirements.txt

format:
# format
    black -l 100 src/*.py

lint:
# linting
    pylint --disable=C0301,C0413,C0103 src/*.py
	
precommit : format lint
