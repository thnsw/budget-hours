# Billable Hours Classifier

This project extracts time entries from Microsoft SQL Server and uses Azure OpenAI to classify them as billable or non-billable.

## To-do
- [ ] Shorten explanation output from LLM
- [ ] Shorten Employee in 'Disapproved hours detail'
- [ ] Add llm_describer.py that describes project performance

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
- `--summary`: Generate a summary report
- `--summary-output`: Path for the summary report output
- `--limit`: Limit the number of records to process
- `--batch-size`: Batch size for API rate limiting (default: 100)
- `--dry-run`: Run without making actual API calls (for testing)
- `--evaluate`: Print detailed evaluation metrics comparing predictions with actual values

## Docker Configuration

The default command in the docker-compose.yml runs with the following options:
```
--summary --output /app/output/classified_hours.csv --summary-output /app/output/summary_report.txt
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

1. Extracts timesheet data from a SQL Server database, specifically focusing on February 2025 data
2. Processes each entry through Azure OpenAI to classify it as billable or non-billable
3. Compares LLM predictions with actual values from the database
4. Evaluates classification performance against human-approved entries
5. Outputs the classified data to a CSV file and optional summary report with performance metrics
6. Supports command-line arguments for flexible execution

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

Add a summary report to any run:

```
python src/main.py --summary
```

### Evaluate Performance

Print detailed classification performance metrics:

```
python src/main.py --evaluate
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
| `--summary` | Generate a summary report |
| `--summary-output PATH` | Path for summary report output |
| `--limit N` | Limit number of records processed |
| `--batch-size N` | Batch size for API rate limiting (default: 100) |
| `--dry-run` | Run without making actual API calls |
| `--evaluate` | Print detailed evaluation metrics comparing predictions with actual values |

## Examples

### Process February 2025 data with performance evaluation:
```
python src/main.py --evaluate --summary
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

- `src/data_extraction.py` - Functions to extract data from SQL Server, focusing on February 2025
- `src/llm_classifier.py` - LLM-based classification using Azure OpenAI
- `src/output_generator.py` - CSV output, performance evaluation, and summary report generation
- `src/main.py` - Main CLI runner script

## Performance Metrics

The classification pipeline evaluates performance using the following metrics:

- **Accuracy**: Percentage of correctly classified entries
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Human-LLM Agreement**: Percentage of entries where LLM classification matches human approval

The summary report includes a confusion matrix and detailed project-by-project breakdown comparing predicted vs. actual billable hours.

## Data Extraction Details

The pipeline extracts the following fields from the SQL Server database:
- Hours
- Date
- Project Name
- Customer Name
- Organization Name
- Description
- Employee Name
- Actual billable status (for evaluation)
- Approval status (for human vs. LLM comparison)

## Development

To create a development environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## License

[Specify your license]
