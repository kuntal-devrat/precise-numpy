# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities privately by opening a GitHub advisory
at:

https://github.com/kuntal-devrat/precise-numpy/security/advisories/new

or by emailing the maintainers. Please do **not** open a public issue for
security-related problems.

## Scope

This project guarantees rigorous error bounds for floating-point interval
arithmetic. Bugs that silently **understate** an error radius or that allow
arbitrary code execution (e.g. via unsafe deserialization) are treated as
high-priority security issues. Bugs that merely overstate a radius are
correctness issues, not security issues.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |
