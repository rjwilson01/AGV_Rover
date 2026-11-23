# Convenience and build targets. Works with GNU make (winget install ezwinports.make) and Linux/macOS.
# Usage: make        (builds out-of-date docs, compiles, and runs selfcheck)
#        make help   (lists all available targets)
PYTHON ?= python
.DEFAULT_GOAL := all

README_DEPS := make_readme.py cli.py $(wildcard configs/*.toml configs/test/*.toml)

.PHONY: help all docs readme diagrams compile selfcheck view clean-logs

help:
	@echo Targets:
	@echo   make            - default: run all (docs, compile, selfcheck)
	@echo   make all        - regenerate out-of-date docs, compile, and run selfcheck
	@echo   make docs       - regenerate README.md and drawio diagrams if out of date
	@echo   make readme     - regenerate README.md if script, cli.py, or configs changed
	@echo   make diagrams   - regenerate docs/architecture.drawio and docs/hitch_forces.drawio
	@echo   make compile    - byte-compile every .py file to catch syntax errors
	@echo   make selfcheck  - run selfcheck.py
	@echo   make view       - open the manual sandbox (python main.py VariableDrawbarController --view)
	@echo   make clean-logs - delete sim_log.csv (sim_test_log.csv is kept)
	@echo Override Python interpreter with: make PYTHON=py all

all: docs compile selfcheck

docs: README.md docs/architecture.drawio docs/hitch_forces.drawio

# Phony shortcuts that delegate to the real file targets
readme: README.md
diagrams: docs/architecture.drawio docs/hitch_forces.drawio

# Real file targets (only run when inputs are newer than the output)
README.md: $(README_DEPS)
	$(PYTHON) make_readme.py

docs/architecture.drawio docs/hitch_forces.drawio: make_diagram.py
	$(PYTHON) make_diagram.py

# Phony validation & convenience targets
compile:
	$(PYTHON) -m compileall -q .

selfcheck:
	$(PYTHON) selfcheck.py

view: compile selfcheck
	$(PYTHON) main.py VariableDrawbarController --view

clean-logs:
	$(PYTHON) -c "import os; os.path.exists('sim_log.csv') and os.remove('sim_log.csv')"
