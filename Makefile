RUSTC ?= rustc
BIN := target/terasweep
SRC := src/sweep.rs src/results.rs src/ridge.rs src/dlinear.rs src/lightgbm.rs src/tide.rs
TRAIN := data/train.bin
TEST := data/test.bin
PRICE := data/price.bin
CONFIG ?= configs/sweep.json
RUNS ?= 1000
OUT ?= runs/$(RUNS)

RUSTFLAGS := --edition=2021 -O -C panic=abort -C target-cpu=native -C lto=fat -C codegen-units=1

.PHONY: build smoke run verify clean

build: $(BIN)

$(BIN): $(SRC)
	mkdir -p target
	$(RUSTC) src/sweep.rs $(RUSTFLAGS) -o $(BIN)

smoke:
	$(MAKE) --no-print-directory run RUNS=1000 OUT=runs/smoke

run: build
	mkdir -p $(OUT)
	/usr/bin/time -p $(BIN) $(TRAIN) $(TEST) $(PRICE) $(CONFIG) $(RUNS) $(OUT)/summary.json 2> $(OUT)/time.txt

verify: clean
	$(MAKE) --no-print-directory run RUNS=1000000 OUT=runs/verify-1m
	python3 scripts/compare_golden.py golden/1m.json runs/verify-1m/summary.json

clean:
	rm -rf target runs
