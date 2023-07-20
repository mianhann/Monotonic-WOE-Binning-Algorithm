# run everything in the same shell
.ONESHELL:


NAME_ = Monotonic-Woe-Binning
ENV_ = py311_env
SHELL = /bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


.DEFAULT_GOAL := help
.PHONY: help
help:
	@echo " "
	@echo "Welcome to $(NAME_)!"
	@echo "================================="
	@echo "Type help to see instructions"
	@echo "Type activate_env to see activate conda environment"
	@echo "Type sort_dependencies to sort dependencies"
	@echo "Type black_linter to use black on .py files"
	@echo "Type test to run unit tests"
	@echo " "


.PHONY: activate_env
activate_env:
	$(CONDA_ACTIVATE) $(ENV_);


.PHONY: sort_dependencies
sort_dependencies:
	$(CONDA_ACTIVATE) $(ENV_); \
	isort . 


.PHONY: black_linter
black_linter:
	$(CONDA_ACTIVATE) $(ENV_); \
	black . 


.PHONY: test
test:
	$(CONDA_ACTIVATE) $(ENV_); \
	pytest --verbose
