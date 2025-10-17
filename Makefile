NAME := rag-llm-demo-ui
TAG := latest
CONTAINER ?= $(NAME):$(TAG)
REGISTRY ?= localhost
UPLOADREGISTRY ?= quay.io/validatedpatterns

.DEFAULT_GOAL := help

##@ Help-related tasks
.PHONY: help
help: ## Help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^(\s|[a-zA-Z_0-9-])+:.*?##/ { printf "  \033[36m%-35s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Local tasks

.PHONY: run
run: ## Run this application
	python app.py

.PHONY: update-deps
update-deps: ## Update the dependencies
	pip-compile --upgrade

.PHONY: install-deps
install-deps: ## Install the dependencies
	pip-sync requirements.txt

##@ Container tasks

.PHONY: build
build: ## Build the container
	podman build -t $(REGISTRY)/$(CONTAINER) .

.PHONY: upload
upload: ## Push the container to the upload registry
	podman tag $(REGISTRY)/$(CONTAINER) $(UPLOADREGISTRY)/$(CONTAINER)
	podman push $(UPLOADREGISTRY)/$(CONTAINER)

##@ Test tasks

.PHONY: super-linter
super-linter: ## Runs super linter locally
	rm -rf .mypy_cache
	podman run -e RUN_LOCAL=true -e USE_FIND_ALGORITHM=true	\
					$(DISABLE_LINTERS) \
					-e VALIDATE_ENV=false \
					-e VALIDATE_PYTHON_ISORT=false \
					-e VALIDATE_PYTHON_RUFF_FORMAT=false \
					-e VALIDATE_TRIVY=false \
					-v $(PWD):/tmp/lint:rw,z \
					-w /tmp/lint \
					ghcr.io/super-linter/super-linter:slim-v8
