RUSTC ?= rustc
CARGO ?= cargo
CARGO_TARGET_DIR ?= target
# Keep target/terasweep available as the historical executable, when present.
BIN ?= target/terasweep-funnel
TRAIN ?= data/train.bin
TEST ?= data/test.bin
PRICE ?= data/price.bin
CONFIG ?= configs/sweep.json
LEVEL ?= compact
RUNS ?= 1000
OUT ?= runs/$(RUNS)

CARGO_BUILD = CARGO_TARGET_DIR="$(abspath $(CARGO_TARGET_DIR))" RUSTC="$(RUSTC)" RUSTFLAGS="$(RUSTFLAGS) -C target-cpu=native" $(CARGO)
RELEASE_BIN := $(CARGO_TARGET_DIR)/release/terasweep

.PHONY: build smoke run verify test clean

build:
	$(CARGO_BUILD) build --release --locked --bin terasweep
	mkdir -p $(dir $(BIN))
	@if ! cmp -s "$(RELEASE_BIN)" "$(BIN)"; then \
		out_bin=$$(mktemp "$(BIN).XXXXXX") || exit; \
		trap 'rm -f "$$out_bin"' EXIT; \
		cp -p "$(RELEASE_BIN)" "$$out_bin" && mv -f "$$out_bin" "$(BIN)"; \
	fi

smoke: build
	@mkdir -p runs; out=$$(mktemp -d runs/smoke-XXXXXX); \
	$(MAKE) --no-print-directory run RUNS=1000 OUT=$$out

run: build
	mkdir -p $(OUT)
	TERASWEEP_LEVEL=$(LEVEL) /usr/bin/time -p $(BIN) $(TRAIN) $(TEST) $(PRICE) $(CONFIG) $(RUNS) $(OUT)/summary.json 2> $(OUT)/time.txt

verify: build
	@mkdir -p runs; out=$$(mktemp -d runs/verify-1m-XXXXXX); \
	$(MAKE) --no-print-directory run RUNS=1000000 OUT=$$out && \
	python3 scripts/compare_golden.py golden/1m.json $$out/summary.json

test: build
	$(CARGO_BUILD) test --locked --lib -- --nocapture
	TERASWEEP_TEST_BINARY="$(abspath $(BIN))" python3 -B tests/test_model_run.py

clean:
	rm -f $(BIN) "$(RELEASE_BIN)" target/terasweep-tests
