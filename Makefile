.PHONY: all install install-dev install-bench test test-fast compile lint clean smoke status \
        bench fused-bench speed-matrix memory-bench cartesian-bench promote release

all: install-dev test

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-bench:
	pip install -e ".[all]"

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x

compile:
	python -m compileall rfsn_v11 tests scripts benchmarks

lint:
	python -m compileall rfsn_v11 tests scripts benchmarks

bench:
	python benchmarks/run_dense_vs_turbopolar.py --model $(MODEL)

fused-bench:
	python benchmarks/run_fused_forced_decode.py --model $(MODEL)

speed-matrix:
	python benchmarks/run_speed_matrix.py --model $(MODEL)

memory-bench:
	python benchmarks/run_memory_bench.py

cartesian-bench:
	python benchmarks/run_cartesian_int8_baseline.py --model $(MODEL)

promote:
	python scripts/run_promotion_suite.py --model $(MODEL)

fast-bench:
	python benchmarks/run_fast_attention_bench.py --model $(MODEL)

smoke:
	python scripts/smoke_test.py
	python scripts/readme_example.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build dist *.egg-info

status:
	@echo "=== TurboPolar repository status ==="
	@echo "Version: $$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])' 2>/dev/null || echo 'unknown')"
	@echo "MLX:     $$(python -c 'import mlx.core as mx; print(mx.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "Tests:   run \`make test\`"

release:
	python scripts/release.py
