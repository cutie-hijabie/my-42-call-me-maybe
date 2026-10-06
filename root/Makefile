.PHONY: all install run debug lint lint-strict clean fclean re

all: install run

install:
	@echo "==> Creating virtual environment and Installing dependencies useing uv sync"
	@uv sync

run:
	@echo "==> Running"
	@uv run python -m src

debug:
	@echo "==> Starting in debug mode..."
	@uv run python -m pdb -m src

lint:
	@echo "==> Running flake8..."
	@flake8 . --extend-exclude=llm_sdk,tests,moulinette,.venv || true
	@echo "==> Running mypy..."
	@mypy .\
		--warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs\
		--exclude '(^|/)(llm_sdk|tests|moulinette)/' \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs || true
		
	@echo "==> Lint complete."

lint-strict:
	@echo "==> Running flake8 (strict)..."
	@flake8 . || true
	@echo "==> Running mypy (strict)..."
	@mypy . --strict || true
	@echo "==> Strict lint complete."

clean:
	@echo "==> Cleaning caches..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "==> Clean complete."

fclean: clean
	@echo "==> Removing virtual environment..."
	@rm -rf .venv
	@echo "==> Full clean complete."

re: fclean all

