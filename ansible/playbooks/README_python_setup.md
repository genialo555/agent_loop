# Python 3.13.5 Setup Playbook

This playbook configures a complete Python 3.13.5 development environment using pyenv with strict typing support.

## Features

- **Python 3.13.5** installation via pyenv with optimizations
- **Virtual environment** management with pyenv-virtualenv
- **Type checking tools**: mypy, pyright, ruff
- **Code formatting**: black
- **Testing framework**: pytest with coverage
- **Modern Python packages**: pydantic v2, attrs, typing-extensions
- **VS Code configuration** for optimal Python development

## Prerequisites

- Ubuntu 24.04 LTS
- User 'agent' must exist on the target system
- Ansible installed on the control machine

## Usage

```bash
ansible-playbook -i inventory python-setup.yml
```

## What it configures

1. **System Dependencies**: All required libraries for building Python 3.13.5
2. **pyenv**: Python version management tool
3. **Python 3.13.5**: Built with optimizations and LTO
4. **Virtual Environment**: Global virtual environment named 'gemma-agent'
5. **Development Tools**:
   - black: Code formatter
   - ruff: Fast Python linter
   - mypy: Static type checker
   - pyright: Microsoft's type checker
   - pytest: Testing framework
   - pydantic: Data validation library
6. **Configuration Files**:
   - `.python-version`: Sets Python 3.13.5 as default
   - `pyproject.toml`: Strict typing configuration
   - `.vscode/settings.json`: VS Code Python settings
   - `type_check.py`: Utility script for running all type checkers

## Type Checking

The playbook configures strict type checking with:

- mypy in strict mode
- All functions must have type annotations
- No implicit Optional types
- Warnings for unused ignores and redundant casts
- Strict equality checking

Run type checks with:
```bash
python ~/type_check.py
```

## Environment Variables

After installation, pyenv will be configured in `.bashrc` with:
- `PYENV_ROOT`: Set to `/home/agent/.pyenv`
- `PATH`: Updated to include pyenv binaries
- pyenv initialization commands

## Python Build Optimizations

Python 3.13.5 is built with:
- `--enable-optimizations`: Profile-guided optimizations
- `--with-lto`: Link-time optimization
- `--enable-loadable-sqlite-extensions`: SQLite extension support
- `-march=native -O3`: CPU-specific optimizations

## Verification

After running the playbook, verify the installation:

```bash
python --version  # Should show Python 3.13.5
pyenv version     # Should show the active version
pip list          # Should show all installed packages
```

## Example Code

See `/home/jerem/agent_loop/example_typed_code.py` for a comprehensive example of properly typed Python code following all modern best practices.