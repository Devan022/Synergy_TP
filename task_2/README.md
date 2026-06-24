# Task 2 - Python Recap Assignment

## Description
This task reads student submission data from a CSV file, analyzes it, and writes a summary report to a JSON file.

## Folder Contents
- `data/submissions.csv` - Input CSV file containing student submission data
- `src/analyzer.py` - Contains helper functions for reading, analyzing, and writing data
- `src/main.py` - Main program file that runs the analysis
- `output/summary.json` - Generated summary report in JSON format
- `requirements.txt` - Task 2 requirements file

## Features
The program performs the following operations:
- Reads student submission data from a CSV file
- Calculates total number of students
- Calculates submitted and missing submission counts
- Calculates average score
- Finds highest scorer
- Finds lowest scorer among submitted students
- Calculates domain-wise average score
- Lists students who did not submit
- Lists students scoring below 5
- Writes all results to a JSON file

## Error Handling
The program handles:
- Missing input CSV files
- Invalid score values in the CSV
- Empty CSV files
- Missing output folder (it creates the folder automatically if needed)

## Run Command

From the repository root, run:

```bash
python task_2/src/main.py task_2/data/submissions.csv task_2/output/summary.json