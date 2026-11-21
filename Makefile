# Convenience targets. Works with GNU make from `winget install ezwinports.make` (cmd.exe) and on Linux/macOS.
# Usage: make help
PYTHON ?= python

.PHONY: help docs readme diagrams compile selfcheck view clean-logs

help:
	@echo Targets:
	@echo   make docs       - regenerate README auto sections AND the drawio diagrams
	@echo   make readme     - regenerate README.md auto sections only (python make_readme.py)
	@echo   make diagrams   - regenerate docs/architecture.drawio and docs/hitch_forces.drawio
	@echo   make compile    - byte-compile every .py file to catch syntax errors
	@echo   make selfcheck  - run selfcheck.py
	@echo   make view       - open the manual sandbox (python main.py VariableDrawbarController --view)
	@echo   make clean-logs - delete sim_log.csv (sim_test_log.csv is kept)
	@echo Override the interpreter with: make PYTHON=py docs

docs: readme diagrams

readme:
	$(PYTHON) make_readme.py

diagrams:
	$(PYTHON) make_diagram.py

compile:
	$(PYTHON) -m compileall -q .

selfcheck:
	$(PYTHON) selfcheck.py

view:
	$(PYTHON) main.py VariableDrawbarController --view

clean-logs:
	$(PYTHON) -c "import os; os.path.exists('sim_log.csv') and os.remove('sim_log.csv')"
