RUSTC ?= rustc
# Keep target/terasweep available as the historical executable, when present.
BIN ?= target/terasweep-funnel
SRC := $(wildcard src/*.rs) $(wildcard src/optimized/*.rs) $(wildcard src/optimized/*.S)
TRAIN ?= data/train.bin
TEST ?= data/test.bin
PRICE ?= data/price.bin
CONFIG ?= configs/sweep.json
LEVEL ?= compact
RUNS ?= 1000
OUT ?= runs/$(RUNS)

RUSTFLAGS := --edition=2021 -O -C panic=abort -C target-cpu=native -C lto=fat -C codegen-units=1

.PHONY: build smoke run verify test clean

build: $(BIN)

$(BIN): $(SRC)
	mkdir -p $(dir $(BIN))
	$(RUSTC) src/sweep.rs $(RUSTFLAGS) -o $(BIN)

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

test:
	mkdir -p target
	$(RUSTC) --test src/sweep.rs --edition=2021 -O -C target-cpu=native -o target/terasweep-tests
	./target/terasweep-tests --nocapture

clean:
	rm -f $(BIN) target/terasweep-tests
