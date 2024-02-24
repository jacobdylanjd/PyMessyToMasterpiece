.PHONY: clean-build
SHELL=/bin/bash

# Delete all build artifacts:
clean-build-project:
	rm -fr packages/project_package/build/
	rm -fr packages/project_package/dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-build-shared:
	rm -fr packages/shared_package/build/
	rm -fr packages/shared_package/dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

# Package project:
package-project: clean-build-project
	python3 packages/project_package/setup.py bdist_wheel
	mkdir -p packages/wheels
	mv packages/project_package/dist/*.whl packages/wheels/
	$(MAKE) clean-build-project

# Package shared:
package-shared: clean-build-shared
	python3 packages/shared_package/setup.py bdist_wheel
	mkdir -p packages/wheels
	mv packages/shared_package/dist/*.whl packages/wheels/
	$(MAKE) clean-build-shared

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test: test-unit test-integration