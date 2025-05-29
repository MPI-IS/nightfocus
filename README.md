# NightFocus

[![Tests](https://github.com/yourusername/nightfocus/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/nightfocus/actions/workflows/tests.yml)

A Python library for focus optimization in astrophotography.

## Features

- Focus optimization using various focus metrics
- Support for both simulated and real cameras
- Dataset generation and management
- Comprehensive test suite

## CI/CD Status

This project uses GitHub Actions for continuous integration. Every push to `master` and every pull request triggers the test suite, which runs:

- Unit tests with pytest
- Compatibility testing across Python 3.8, 3.9, 3.10, and 3.11

## Installation

```bash
pip install -e .[dev]
```

## Running Tests

```bash
pytest tests/
```

## Development

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## License

MIT