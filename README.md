# Climate Challenge (African Climate Trend Analysis) – Environment Setup

## Overview

This repository contains the environment setup for the Climate Challenge project.
It demonstrates how to create a reproducible Python environment using `venv`, manage dependencies with `requirements.txt`, and automate installation using a GitHub Actions workflow.

---

## Prerequisites

Make sure the following tools are installed on your system:

* Python 3.11 (or compatible version)
* Git
* A terminal (Command Prompt, PowerShell, or Bash)

Check your Python version:

```bash
python --version
```

---

## Clone the Repository

Clone the project to your local machine:

```bash
git clone https://github.com/soliana-k/climate-challenge-week0/
cd climate-challenge-week0
```

---

## Create a Virtual Environment

Create a virtual environment using `venv`:

```bash
python -m venv venv
```

Activate the virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

You should see `(venv)` appear in your terminal.

---

## Install Dependencies

Upgrade pip and install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

All necessary libraries will be installed automatically.

---

## Reproducing the Environment

To reproduce the environment on a new machine:

1. Clone the repository
2. Create and activate the virtual environment
3. Install dependencies from `requirements.txt`

These steps ensure the same software environment can be recreated consistently.

---

## Continuous Integration (CI)

This project includes a GitHub Actions workflow that automatically:

* Sets up Python
* Installs dependencies
* Verifies the environment setup

The workflow runs on:

* Push to the `setup-task` branch
* Pull requests to the `main` branch

---

## Project Structure

```
climate-challenge-week0/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── notebooks/
|   └── compare_countries.ipynb
|   └── ethiopia_eda.ipynb
|   └── kenya_eda.ipynb
|   └── nigeria_eda.ipynb
|   └── sudan_eda.ipynb
|   └── tanzania_eda.ipynb
|
|
├── src/
|   └── __init__.py
|   └── data_processor.py
|
|   
├── requirements.txt
├── README.md
├── .gitignore
└── 
```

---

## Notes

* The `venv/` directory is excluded from version control using `.gitignore`.
* Dependencies are managed through `requirements.txt`.
* API keys and sensitive information should be stored in a `.env` file and ignored by Git.

---

## Author

Kalkidan Kassahun

