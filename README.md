# Hour Approval Predictor

This project extracts time entries from Microsoft SQL Server, uses Azure OpenAI to predict whether each entry should be approved, and generates a PDF summary report to assist team leads and financial teams in reviewing registered hours.

## To-do
- [X] Shorten explanation output from LLM
- [X] Shorten Employee in 'Disapproved hours detail'
- [X] Add llm_describer.py that describes project performance
- [X] Add customer llm call to describ customer performance
- [ ] Add automated prompt optimization to generate prompt (https://arxiv.org/abs/2305.03495 https://github.com/Eladlev/AutoPrompt)
- [ ] Add table of contents with the following structure: Overall and customer are first in the hierarchy. Projects is a sub-section below each customer.
- [ ] Update overall statistics.



## Maybe-do

## Setup

1. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

2. Create an output directory:
   ```bash
   mkdir -p output
   ```

3. Build and run the Docker container using docker-compose:
   ```bash
   docker-compose build
   docker-compose up
   ```

## Configuration

The application takes several command-line arguments:

- `--extract-only`: Only extract data from SQL Server without classification
- `--classify-only`: Only classify data (requires input file with `--input`)
- `--input`: Input CSV file path (for `--classify-only` mode)
- `--output`: Output CSV file path
- `--summary`: Generate a PDF summary report
- `--summary-output`: Path for the PDF summary report output (default: output/summary_report_YYYYMMDD_HHMMSS.pdf)
- `--limit`: Limit the number of records to process
- `--batch-size`: Batch size for API rate limiting (default: 100)
- `--dry-run`: Run without making actual API calls (for testing)

## Docker Configuration

The default test command with limit and summary:
```bash
docker-compose run --rm billable-hours-classifier --limit 1 --summary
```

You can modify these options in the docker-compose.yml file to suit your needs.

## Example Usage

To run with different parameters:

```bash
docker-compose run --rm billable-hours-classifier --limit 100 --dry-run
```

To extract data only (no classification):

```bash
docker-compose run --rm billable-hours-classifier --extract-only --output /app/output/extracted_data.csv
```

## Environment Variables

- `SQL_SERVER`: SQL Server hostname
- `SQL_DATABASE`: Database name
- `SQL_USERNAME`: SQL Server username
- `SQL_PASSWORD`: SQL Server password
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_API_VERSION`: Azure OpenAI API version
- `AZURE_OPENAI_DEPLOYMENT`: Azure OpenAI deployment name

## Overview

This application consists of a modular pipeline that:

1. Extracts timesheet data from a SQL Server database (`data_extraction.py`).
2. Processes each entry through Azure OpenAI to predict its approval status (`llm_classifier.py`).
3. Generates descriptive summaries for projects and customers using Azure OpenAI (`llm_describer.py`).
4. Outputs the detailed classified data to a CSV file (`output_generator.py`).
5. Generates a comprehensive PDF summary report including overall statistics, project/customer descriptions, performance metrics, and detailed hour breakdowns (`output_generator.py`).
6. Supports command-line arguments for flexible execution (`main.py`).

## Prerequisites

- Python 3.8+
- SQL Server with ODBC Driver 18
- Azure OpenAI API access

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file based on `.env.example`:
```
cp .env.example .env
```

4. Edit `.env` with your SQL Server and Azure OpenAI credentials:
```
# SQL Server connection
SQL_SERVER=your_server_name
SQL_DATABASE=your_database_name
SQL_USERNAME=your_username
SQL_PASSWORD=your_password

# Azure OpenAI connection
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

## Usage

The application can be run in several modes:

### Full Pipeline

Process data from SQL Server, classify it, and output to CSV:

```
python src/main.py
```

### Extract Only

Extract data from SQL Server without classification:

```
python src/main.py --extract-only
```

### Classify Only

Classify data from a CSV file (previously extracted):

```
python src/main.py --classify-only --input path/to/input.csv
```

### Generate Summary

Add a PDF summary report to any run:

```
python src/main.py --summary
```

### Dry Run

Test the pipeline without making actual API calls:

```
python src/main.py --dry-run
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--extract-only` | Only extract data, no classification |
| `--classify-only` | Only classify data (requires `--input`) |
| `--input PATH` | Input CSV file path |
| `--output PATH` | Output CSV file path |
| `--summary` | Generate a PDF summary report |
| `--summary-output PATH` | Path for PDF summary report output |
| `--limit N` | Limit number of records processed |
| `--batch-size N` | Batch size for API rate limiting (default: 100) |
| `--dry-run` | Run without making actual API calls |

## Examples

### Process data and generate CSV + PDF Summary:
```
python src/main.py --summary
```

### Process data from a CSV file:
```
python src/main.py --classify-only --input data/hours.csv --output results/classified.csv
```

### Extract data only with custom output path:
```
python src/main.py --extract-only --output data/extracted_hours.csv
```

### Process with smaller batch size (for rate limiting):
```
python src/main.py --batch-size 5 --summary
```

## Project Structure

- `src/data_extraction.py` - Functions to extract data from SQL Server.
- `src/llm_classifier.py` - LLM-based classification to predict approval status using Azure OpenAI.
- `src/llm_describer.py` - LLM-based generation of project and customer descriptions using Azure OpenAI.
- `src/output_generator.py` - Handles CSV output generation and PDF summary report creation (including performance evaluation).
- `src/main.py` - Main CLI runner script, orchestrates the pipeline.
- `src/sw-theme/` - Contains image assets (header, footer) for the PDF report theme.

## Performance Metrics

The classification pipeline evaluates performance by comparing the predicted approval status (`is_approved_predicted`) with the actual status (`is_approved`) using the following metrics (included in the PDF summary report):

- **Accuracy**: Percentage of correctly classified entries (approved/not approved).
- **Precision**: True positives / (True positives + False positives) - Measures the accuracy of positive predictions.
- **Recall**: True positives / (True positives + False negatives) - Measures how many actual positives were correctly identified.
- **F1 Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Table showing True Positives, True Negatives, False Positives, and False Negatives.

The PDF summary report includes these overall metrics and a breakdown comparing predicted vs. actual approved hours.

## Data Extraction Details

The pipeline extracts the following fields from the SQL Server database:
- Hours
- Date
- Project Name
- Customer Name
- Organization Name
- Description (of the time entry)
- Is Approved (Actual status from database)

## Development

To create a development environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## License

[Specify your license]
