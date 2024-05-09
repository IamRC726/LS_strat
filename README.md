# Readme

## Requirements Management

Uses [Poetry](https://python-poetry.org/) to manage dependencies.

Install Poetry with the command

```
    $ python -m pip install poetry
```

After poetry is installed, additional dependencies from [pyproject.toml](./pyproject.toml) can be installed using

```
    $ poetry install
```

Additional dependencies can be added using

```
    $ poetry add <dependency_name>
```

## Database

This project uses [sqlite](https://www.sqlite.org/), a lightweight relational database to store information
that will be used during analysis.

Please [download](https://www.sqlite.org/download.html) and install the appropriate version for your OS.

The database is stored on disk in a file. This file is configured
through environment variable "DB_LOCATION". See [.env.shared](.env.shared) for its default setting.

## Linting

Uses [Ruff](https://github.com/astral-sh/ruff) for super fast linting of python code.

A pre-commit has been added using ruff, and can be executed using

```
    $ pre-commit install

    or

    $ poetry run pre-commit install
```
