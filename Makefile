.ONESHELL:
.PHONY: test test-complete test-default test-artifacts test-corrections test-external lint lint-v

PYTEST ?= python -m pytest
PYTEST_WORKERS ?= 4

# Complete FELIS unit test workflow. This includes artifact-gated tests and
# the standard FELIS unit suite, both with coverage enabled.
test: test-complete

test-complete: test-artifacts test-default

test-default:
	set -ex
	$(PYTEST) -vv -n $(PYTEST_WORKERS) --dist load --disable-warnings --cov=felis --cov-report=term --cov-config=pyproject.toml --random-order \
		--ignore=felis/tests/artifacts --ignore=felis/tests/testkit \
		felis/tests

test-artifacts:
	set -ex
	FELIS_TEST_ARTIFACTS=1 \
	$(PYTEST) -vv -n $(PYTEST_WORKERS) --dist load --disable-warnings --cov=felis --cov-report=term --cov-config=pyproject.toml --random-order felis/tests/artifacts

test-corrections:
	set -ex
	$(PYTEST) -vv -n $(PYTEST_WORKERS) --dist load --disable-warnings --cov=felis --cov-report=term --cov-config=pyproject.toml --random-order felis/tests/protocols/correction

test-external:
	set -ex
	$(PYTEST) -vv -n $(PYTEST_WORKERS) --dist load --disable-warnings --cov=felis --cov-report=term --cov-config=pyproject.toml --random-order felis/tests/external

lint:
	pylint --rcfile=pyproject.toml --disable=R,C ./felis

lint-v:
	pylint --rcfile=pyproject.toml ./felis
