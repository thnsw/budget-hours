import os
import csv
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

def generate_csv_output(classified_data: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
    """
    Generate a CSV file containing the classified hours data
    
    Args:
        classified_data: List of dictionaries with classified hours data
        output_path: Optional path for the output file. If None, a default path will be generated.
        
    Returns:
        Path to the generated CSV file
    """
    if not classified_data:
        raise ValueError("No data provided for CSV generation")
    
    # Create the output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Generate a default output path if none provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/classified_hours_{timestamp}.csv"
    
    # Define the columns to include in the CSV
    columns = [
        "employee_name",
        "project_name",
        "customer_name",
        "organization_name",
        "hours",
        "description",
        "date",
        "is_approved_predicted",
        "is_approved",
        "classification_confidence",
        "classification_reasoning",
        "prediction_matches_actual"
    ]
    
    # Ensure all keys exist in each entry and add comparison fields
    normalized_data = []
    for entry in classified_data:
        # Calculate match metrics
        if "is_approved" in entry and "is_approved_predicted" in entry and entry["is_approved_predicted"] is not None:
            entry["prediction_matches_actual"] = entry["is_approved_predicted"] == entry["is_approved"]
        else:
            entry["prediction_matches_actual"] = None
            
        normalized_entry = {col: entry.get(col, "") for col in columns}
        normalized_data.append(normalized_entry)
    
    try:
        # Write to CSV using pandas for more control over formatting
        df = pd.DataFrame(normalized_data)
        
        # Reorder columns
        df = df[columns]
        
        # Write to CSV
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        
        print(f"CSV output generated successfully: {output_path}")
        print(f"Total entries: {len(classified_data)}")
        print(f"Approved entries (predicted): {sum(1 for entry in classified_data if entry.get('is_approved_predicted') is True)}")
        print(f"Not Approved entries (predicted): {sum(1 for entry in classified_data if entry.get('is_approved_predicted') is False)}")
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Error generating CSV output: {str(e)}")

def evaluate_classification_performance(classified_data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate the performance of the classifier by comparing predicted values with actual values
    
    Args:
        classified_data: List of dictionaries with classified hours data
        
    Returns:
        Tuple containing (metrics dictionary, confusion matrix DataFrame)
    """
    # Filter out entries where we don't have actual values or predictions
    valid_entries = [entry for entry in classified_data 
                    if "is_approved" in entry         # Changed from is_billable_actual
                    and "is_approved_predicted" in entry # Changed from is_billable_predicted
                    and entry["is_approved_predicted"] is not None]
    
    if not valid_entries:
        return {"error": "No valid entries for evaluation"}, pd.DataFrame()
    
    # Calculate metrics for LLM vs actual values
    y_true = [entry["is_approved"] for entry in valid_entries] # Changed from is_billable_actual
    y_pred = [entry["is_approved_predicted"] for entry in valid_entries] # Changed from is_billable_predicted
    
    # Calculate basic metrics
    true_pos = sum(1 for t, p in zip(y_true, y_pred) if t and p)    # Prediction: Approved, Actual: Approved
    true_neg = sum(1 for t, p in zip(y_true, y_pred) if not t and not p) # Prediction: Not Approved, Actual: Not Approved
    false_pos = sum(1 for t, p in zip(y_true, y_pred) if not t and p)   # Prediction: Approved, Actual: Not Approved
    false_neg = sum(1 for t, p in zip(y_true, y_pred) if t and not p)   # Prediction: Not Approved, Actual: Approved
    
    total = len(valid_entries)
    accuracy = (true_pos + true_neg) / total if total > 0 else 0
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate confusion matrix
    confusion_matrix = pd.DataFrame({
        "Actual Approved": [true_pos, false_neg],
        "Actual Not Approved": [false_pos, true_neg]
    }, index=["Predicted Approved", "Predicted Not Approved"])
    
    metrics = {
        "total_entries_evaluated": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_pos,
        "true_negatives": true_neg,
        "false_positives": false_pos,
        "false_negatives": false_neg
    }
    
    return metrics, confusion_matrix

def generate_summary_report(classified_data: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
    """
    Generate a summary report of the classified hours with performance metrics
    
    Args:
        classified_data: List of dictionaries with classified hours data
        output_path: Optional path for the output file. If None, a default path will be generated.
        
    Returns:
        Path to the generated report file
    """
    if not classified_data:
        raise ValueError("No data provided for summary generation")
    
    # Create the output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Generate a default output path if none provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/summary_report_{timestamp}.txt"
    
    # Calculate performance metrics
    metrics, confusion_matrix = evaluate_classification_performance(classified_data)
    
    # Calculate summary statistics
    total_entries = len(classified_data)
    approved_predicted = sum(1 for entry in classified_data if entry.get('is_approved_predicted') is True)
    not_approved_predicted = sum(1 for entry in classified_data if entry.get('is_approved_predicted') is False)
    error_entries = sum(1 for entry in classified_data if entry.get('is_approved_predicted') is None)
    
    approved_actual = sum(1 for entry in classified_data if entry.get('is_approved') is True)
    not_approved_actual = sum(1 for entry in classified_data if entry.get('is_approved') is False)
    
    total_hours = sum(float(entry.get('hours', 0)) for entry in classified_data)
    approved_hours_predicted = sum(float(entry.get('hours', 0)) for entry in classified_data 
                             if entry.get('is_approved_predicted') is True)
    not_approved_hours_predicted = sum(float(entry.get('hours', 0)) for entry in classified_data 
                                 if entry.get('is_approved_predicted') is False)
    
    # Group by project
    projects = {}
    for entry in classified_data:
        project = entry.get('project_name', 'Unknown')
        if project not in projects:
            projects[project] = {
                'total_hours': 0,
                'approved_hours_predicted': 0,
                'not_approved_hours_predicted': 0,
                'approved_hours_actual': 0,
                'not_approved_hours_actual': 0,
                'entries': 0
            }
        
        projects[project]['entries'] += 1
        hours = float(entry.get('hours', 0))
        projects[project]['total_hours'] += hours
        
        if entry.get('is_approved_predicted') is True:
            projects[project]['approved_hours_predicted'] += hours
        elif entry.get('is_approved_predicted') is False:
            projects[project]['not_approved_hours_predicted'] += hours
            
        if entry.get('is_approved') is True:
            projects[project]['approved_hours_actual'] += hours
        elif entry.get('is_approved') is False:
            projects[project]['not_approved_hours_actual'] += hours
    
    try:
        with open(output_path, 'w') as f:
            f.write("Hours Approval Summary Report\n")
            f.write("=============================\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Classification Performance Metrics (Prediction vs Actual Approval)\n")
            f.write("------------------------------------------------------------\n")
            if "error" not in metrics:
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision (for Approved): {metrics['precision']:.4f}\n")
                f.write(f"Recall (for Approved): {metrics['recall']:.4f}\n")
                f.write(f"F1 Score (for Approved): {metrics['f1_score']:.4f}\n\n")
                
                f.write("Confusion Matrix:\n")
                f.write(f"{confusion_matrix.to_string()}\n\n")
            else:
                f.write(f"Error: {metrics['error']}\n\n")
            
            f.write("Overall Statistics\n")
            f.write("-----------------\n")
            f.write(f"Total Entries Processed: {total_entries}\n")
            f.write(f"Approved Entries (Predicted): {approved_predicted} ({approved_predicted/total_entries*100:.1f}%)\n")
            f.write(f"Not Approved Entries (Predicted): {not_approved_predicted} ({not_approved_predicted/total_entries*100:.1f}%)\n")
            f.write(f"Error Entries (Prediction Failed): {error_entries} ({error_entries/total_entries*100:.1f}%)\n\n")
            
            f.write(f"Approved Entries (Actual): {approved_actual} ({approved_actual/total_entries*100:.1f}%)\n")
            f.write(f"Not Approved Entries (Actual): {not_approved_actual} ({not_approved_actual/total_entries*100:.1f}%)\n\n")
            
            f.write(f"Total Hours Processed: {total_hours:.2f}\n")
            f.write(f"Approved Hours (Predicted): {approved_hours_predicted:.2f}\n")
            f.write(f"Not Approved Hours (Predicted): {not_approved_hours_predicted:.2f}\n\n")
            
            f.write("Statistics by Project\n")
            f.write("---------------------\n")
            for project, stats in projects.items():
                f.write(f"Project: {project}\n")
                f.write(f"  Total Entries: {stats['entries']}\n")
                f.write(f"  Total Hours: {stats['total_hours']:.2f}\n")
                f.write(f"  Approved Hours (Predicted): {stats['approved_hours_predicted']:.2f}\n")
                f.write(f"  Not Approved Hours (Predicted): {stats['not_approved_hours_predicted']:.2f}\n")
                f.write(f"  Approved Hours (Actual): {stats['approved_hours_actual']:.2f}\n")
                f.write(f"  Not Approved Hours (Actual): {stats['not_approved_hours_actual']:.2f}\n\n")
        
        print(f"Summary report generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        raise Exception(f"Error generating summary report: {str(e)}")

if __name__ == "__main__":
    # Simple test with mock data
    mock_data = [
        {
            "employee_name": "John Doe", "project_name": "Project A", "customer_name": "Cust A", 
            "hours": 5, "description": "Work", "date": "2024-01-01",
            "is_approved_predicted": True, "is_approved": True, 
            "classification_confidence": 0.9, "classification_reasoning": "Looks good"
        },
        {
            "employee_name": "Jane Smith", "project_name": "Project B", "customer_name": "Cust B", 
            "hours": 3, "description": "Stuff", "date": "2024-01-02",
            "is_approved_predicted": False, "is_approved": True, 
            "classification_confidence": 0.7, "classification_reasoning": "Needs more info"
        },
        {
            "employee_name": "John Doe", "project_name": "Project A", "customer_name": "Cust A", 
            "hours": 2, "description": "More work", "date": "2024-01-03",
            "is_approved_predicted": True, "is_approved": False, 
            "classification_confidence": 0.8, "classification_reasoning": "Internal task mismatch"
        },
        {
            "employee_name": "Peter Jones", "project_name": "Project B", "customer_name": "Cust B", 
            "hours": 6, "description": "Meeting", "date": "2024-01-04",
            "is_approved_predicted": False, "is_approved": False, 
            "classification_confidence": 0.95, "classification_reasoning": "Not approved task type"
        },
        {
            "employee_name": "Jane Smith", "project_name": "Project A", "customer_name": "Cust A", 
            "hours": 4, "description": "Final checks", "date": "2024-01-05",
            "is_approved_predicted": None, "is_approved": True, # Simulate an error entry
            "classification_confidence": 0, "classification_reasoning": "Error: API Timeout"
        }
    ]
    
    try:
        csv_path = generate_csv_output(mock_data)
        report_path = generate_summary_report(mock_data)
        
        print(f"\nCSV file: {csv_path}")
        print(f"Report file: {report_path}")
        
        # Print report content for review
        with open(report_path, 'r') as f:
            print("\nReport Content:")
            print(f.read())
            
    except Exception as e:
        print(f"Error in test: {str(e)}") 