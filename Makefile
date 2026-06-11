.PHONY: all install install-dev test test-fast compile lint clean status

all: install-dev test

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x

compile:
	python -m compileall rfsn_v11 tests

lint:
	python -m compileall rfsn_v11 tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build dist *.egg-info

status:
	@echo "=== TurboPolar repository status ==="
	@echo "Version: $$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])' 2>/dev/null || echo 'unknown')"
	@echo "MLX:     $$(python -c 'import mlx.core as mx; print(mx.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "Tests:   run \`make test\`"
