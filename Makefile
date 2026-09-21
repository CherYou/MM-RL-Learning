.PHONY: setup setup-verl check-verl smoke-verl data check portability docs check-docs check-math nav smoke logs status stop
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
	$(PY) scripts/audit_tutorials.py
	CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 $(PY) -m pytest -q
	$(PY) scripts/verify_data.py
	.venv/bin/ruff check src scripts tests
portability:
	$(PY) scripts/check_portability.py
docs check-docs:
	$(PY) scripts/check_markdown_math.py
	$(PY) scripts/audit_tutorials.py
	$(PY) scripts/check_navigation.py
check-math:
	$(PY) examples/math/loss_walkthrough.py
	CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 $(PY) -m pytest -q tests/test_loss_math_walkthrough.py
nav:
	$(PY) scripts/build_chapter_registry.py
	$(PY) scripts/sync_navigation.py --write
	$(PY) scripts/check_navigation.py
smoke:
	$(PY) scripts/smoke_all.py
logs:
	$(PY) scripts/services.py start
status:
	$(PY) scripts/services.py status
stop:
	$(PY) scripts/services.py stop
