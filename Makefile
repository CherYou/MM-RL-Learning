.PHONY: setup setup-verl check-verl smoke-verl data check portability docs smoke logs status stop
PY := .venv/bin/python
setup:
	bash scripts/setup.sh
setup-verl:
	bash scripts/setup_verl.sh --cpu
check-verl:
	CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .venv-verl/bin/python -m pytest -q
	CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .venv-verl/bin/python scripts/check_verl_fsdp.py
smoke-verl:
	.venv-verl/bin/python scripts/smoke_verl.py
data:
	$(PY) scripts/prepare_all.py
check:
	$(PY) scripts/check_portability.py
	$(PY) scripts/check_markdown_math.py
	CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 $(PY) -m pytest -q
	$(PY) scripts/verify_data.py
	.venv/bin/ruff check src scripts tests
portability:
	$(PY) scripts/check_portability.py
docs:
	$(PY) scripts/check_markdown_math.py
smoke:
	$(PY) scripts/smoke_all.py
logs:
	$(PY) scripts/services.py start
status:
	$(PY) scripts/services.py status
stop:
	$(PY) scripts/services.py stop
