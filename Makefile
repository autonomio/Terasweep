RUSTC ?= rustc
CARGO ?= cargo
CARGO_TARGET_DIR ?= target
BIN := $(abspath $(CARGO_TARGET_DIR)/release/terasweep)
TRAIN ?= tests/fixtures/concrete/train.bin
VALIDATION ?= tests/fixtures/concrete/validation.bin
PLAN ?= docs/examples/models.txt

CARGO_BUILD = CARGO_TARGET_DIR="$(abspath $(CARGO_TARGET_DIR))" RUSTC="$(RUSTC)" $(CARGO)

.PHONY: build run smoke test clean

build:
	@$(CARGO_BUILD) build --release --locked --bin terasweep

run: build
	@"$(BIN)" sweep "$(TRAIN)" "$(VALIDATION)" "$(PLAN)"

smoke: run

test: build
	$(CARGO_BUILD) test --locked
	TERASWEEP_TEST_BINARY="$(BIN)" python3 -B tests/test_model_run.py

clean:
	rm -f "$(BIN)"
