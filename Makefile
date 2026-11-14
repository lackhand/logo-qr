.PHONY: test

test:
	PYTHONPATH=. python3 -m pytest tests -v
