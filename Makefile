.PHONY: clean-build
SHELL=/bin/bash

# Delete all build artifacts:
clean-build:
	rm -fr packages/project_package/build/
	rm -fr packages/project_package/dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

# Package project:
package: clean-build
	python3 packages/project_package/setup.py bdist_wheel
	mkdir -p packages/wheels
	mv packages/project_package/dist/*.whl packages/wheels/
	$(MAKE) clean-build

