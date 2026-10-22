# Makefile for ISM — Uncertainty-Aware Predictive Data Migration
# Usage: make all     (runs everything end-to-end)
#        make test    (runs all unit tests)
#        make s4      (runs stage 4 only)

PYTHON = .venv\Scripts\python.exe
PYTEST = $(PYTHON) -m pytest

.PHONY: all test s1 s2 s3 s4 s5 s6 s7 final jobs aggregate clean

all: s1 s2 s3 s4 s5 s6 jobs aggregate

test:
	$(PYTEST) tests/ -v -s

s1:
	$(PYTHON) run_s1.py

s2:
	$(PYTHON) run_s2.py

s3:
	$(PYTHON) run_s3.py

s4:
	$(PYTHON) run_s4.py

s5:
	$(PYTHON) run_s5.py

s6:
	$(PYTHON) run_s6.py

s7:
	$(PYTHON) run_s7.py

final:
	$(PYTHON) final_pipeline.py

jobs:
	$(PYTHON) launch_jobs.py 5
	$(PYTHON) run_simpy_jobs.py --seeds 3
	$(PYTHON) run_msrc_jobs.py

aggregate:
	$(PYTHON) aggregate_final.py
	$(PYTHON) demo.py

clean:
	del /q results\*.json results\*.csv 2>nul || true
	del /q plots\*.png 2>nul || true
