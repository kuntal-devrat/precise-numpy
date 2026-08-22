# Makefile for precise-numpy development tasks.

.PHONY: build install develop test lint fmt check release-wheel clean

build:
	maturin build --release

install:
	pip install .

develop:
	maturin develop --release

test: test-rust test-python

test-rust:
	cargo test --all-targets --all-features

test-python:
	python -m unittest tests/python/test_api.py -v

lint:
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
	ruff check python tests

fmt:
	cargo fmt --all
	ruff format python tests

check: lint test

clean:
	cargo clean
	rm -rf dist target/wheels
