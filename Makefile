.PHONY: build-verifier test

build-verifier:
	mkdir -p bin
	/usr/bin/swiftc -O tools/vision_qr_check.swift -o bin/vision_qr_check

test:
	PYTHONPATH=. python3 -m pytest tests -v
