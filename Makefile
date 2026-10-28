# CPU-only conveniences. The GPU pipeline is driven by scripts/ (see README).
PY ?= python

.PHONY: figures figures-tr figures-en test demo
figures:      ## rebuild all figures (TR + EN) from results/
	PYTHONIOENCODING=utf-8 $(PY) -m analyze.make_figures
figures-tr:
	PYTHONIOENCODING=utf-8 $(PY) -m analyze.make_figures --lang tr
figures-en:
	PYTHONIOENCODING=utf-8 $(PY) -m analyze.make_figures --lang en
test:         ## unit tests (needs torch CPU)
	PYTHONIOENCODING=utf-8 sh -c 'for t in test_prf test_sampler_detect test_reading_scorer test_boxed_parser test_manifest test_think_ban test_vllm_adapter; do $(PY) -m wm.tests.$$t || exit 1; done'
demo:         ## watermark a toy model and detect it
	PYTHONIOENCODING=utf-8 $(PY) -m examples.toy_demo
