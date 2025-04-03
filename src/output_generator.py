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
        "is_billable_predicted",
        "is_billable_actual",
        "is_approved",
        "classification_confidence",
        "classification_reasoning",
        "prediction_matches_actual",
        "prediction_matches_approved"
    ]
    
    # Ensure all keys exist in each entry and add comparison fields
    normalized_data = []
    for entry in classified_data:
        # Calculate match metrics
        if "is_billable_actual" in entry and "is_billable_predicted" in entry:
            entry["prediction_matches_actual"] = entry["is_billable_predicted"] == entry["is_billable_actual"]
        else:
            entry["prediction_matches_actual"] = None
            
        if "is_approved" in entry and "is_billable_predicted" in entry:
            entry["prediction_matches_approved"] = entry["is_billable_predicted"] == entry["is_approved"]
        else:
            entry["prediction_matches_approved"] = None
            
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
        print(f"Billable entries (predicted): {sum(1 for entry in classified_data if entry.get('is_billable_predicted') is True)}")
        print(f"Non-billable entries (predicted): {sum(1 for entry in classified_data if entry.get('is_billable_predicted') is False)}")
        
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
                    if "is_billable_actual" in entry 
                    and "is_billable_predicted" in entry 
                    and entry["is_billable_predicted"] is not None]
    
    if not valid_entries:
        return {"error": "No valid entries for evaluation"}, pd.DataFrame()
    
    # Calculate metrics for LLM vs actual values
    y_true = [entry["is_billable_actual"] for entry in valid_entries]
    y_pred = [entry["is_billable_predicted"] for entry in valid_entries]
    
    # Calculate basic metrics
    true_pos = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    true_neg = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    false_pos = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    false_neg = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    
    total = len(valid_entries)
    accuracy = (true_pos + true_neg) / total if total > 0 else 0
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate confusion matrix
    confusion_matrix = pd.DataFrame({
        "Actual Billable": [true_pos, false_neg],
        "Actual Non-Billable": [false_pos, true_neg]
    }, index=["Predicted Billable", "Predicted Non-Billable"])
    
    # Calculate metrics for comparing with human approved entries
    valid_approved_entries = [entry for entry in classified_data
                             if "is_approved" in entry
                             and "is_billable_predicted" in entry
                             and entry["is_billable_predicted"] is not None]
    
    if valid_approved_entries:
        y_approved = [entry["is_approved"] for entry in valid_approved_entries]
        y_pred_approved = [entry["is_billable_predicted"] for entry in valid_approved_entries]
        
        approved_agreement = sum(1 for a, p in zip(y_approved, y_pred_approved) if a == p) / len(valid_approved_entries)
    else:
        approved_agreement = None
    
    metrics = {
        "total_entries": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_pos,
        "true_negatives": true_neg,
        "false_positives": false_pos,
        "false_negatives": false_neg,
        "human_agreement": approved_agreement
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
    billable_predicted = sum(1 for entry in classified_data if entry.get('is_billable_predicted') is True)
    non_billable_predicted = sum(1 for entry in classified_data if entry.get('is_billable_predicted') is False)
    error_entries = sum(1 for entry in classified_data if entry.get('is_billable_predicted') is None)
    
    billable_actual = sum(1 for entry in classified_data if entry.get('is_billable_actual') is True)
    non_billable_actual = sum(1 for entry in classified_data if entry.get('is_billable_actual') is False)
    
    total_hours = sum(float(entry.get('hours', 0)) for entry in classified_data)
    billable_hours_predicted = sum(float(entry.get('hours', 0)) for entry in classified_data 
                             if entry.get('is_billable_predicted') is True)
    non_billable_hours_predicted = sum(float(entry.get('hours', 0)) for entry in classified_data 
                                 if entry.get('is_billable_predicted') is False)
    
    # Group by project
    projects = {}
    for entry in classified_data:
        project = entry.get('project_name', 'Unknown')
        if project not in projects:
            projects[project] = {
                'total_hours': 0,
                'billable_hours_predicted': 0,
                'non_billable_hours_predicted': 0,
                'billable_hours_actual': 0,
                'non_billable_hours_actual': 0,
                'entries': 0
            }
        
        projects[project]['entries'] += 1
        hours = float(entry.get('hours', 0))
        projects[project]['total_hours'] += hours
        
        if entry.get('is_billable_predicted') is True:
            projects[project]['billable_hours_predicted'] += hours
        elif entry.get('is_billable_predicted') is False:
            projects[project]['non_billable_hours_predicted'] += hours
            
        if entry.get('is_billable_actual') is True:
            projects[project]['billable_hours_actual'] += hours
        elif entry.get('is_billable_actual') is False:
            projects[project]['non_billable_hours_actual'] += hours
    
    try:
        with open(output_path, 'w') as f:
            f.write("Hours Classification Summary Report\n")
            f.write("================================\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Classification Performance Metrics\n")
            f.write("--------------------------------\n")
            if "error" not in metrics:
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
                f.write(f"Human Agreement: {metrics['human_agreement']:.4f}\n\n" if metrics['human_agreement'] is not None else "Human Agreement: N/A\n\n")
                
                f.write("Confusion Matrix:\n")
                f.write(f"{confusion_matrix.to_string()}\n\n")
            else:
                f.write(f"Error: {metrics['error']}\n\n")
            
            f.write("Overall Statistics\n")
            f.write("-----------------\n")
            f.write(f"Total Entries: {total_entries}\n")
            f.write(f"Billable Entries (Predicted): {billable_predicted} ({billable_predicted/total_entries*100:.1f}%)\n")
            f.write(f"Non-Billable Entries (Predicted): {non_billable_predicted} ({non_billable_predicted/total_entries*100:.1f}%)\n")
            f.write(f"Error Entries: {error_entries} ({error_entries/total_entries*100:.1f}%)\n\n")
            
            f.write(f"Billable Entries (Actual): {billable_actual} ({billable_actual/total_entries*100:.1f}%)\n")
            f.write(f"Non-Billable Entries (Actual): {non_billable_actual} ({non_billable_actual/total_entries*100:.1f}%)\n\n")
            
            f.write(f"Total Hours: {total_hours:.2f}\n")
            f.write(f"Billable Hours (Predicted): {billable_hours_predicted:.2f} ({billable_hours_predicted/total_hours*100:.1f}%)\n")
            f.write(f"Non-Billable Hours (Predicted): {non_billable_hours_predicted:.2f} ({non_billable_hours_predicted/total_hours*100:.1f}%)\n\n")
            
            f.write("Project Breakdown\n")
            f.write("----------------\n")
            for project, stats in projects.items():
                f.write(f"\nProject: {project}\n")
                f.write(f"  Entries: {stats['entries']}\n")
                f.write(f"  Total Hours: {stats['total_hours']:.2f}\n")
                
                f.write(f"  Billable Hours (Predicted): {stats['billable_hours_predicted']:.2f} ")
                if stats['total_hours'] > 0:
                    f.write(f"({stats['billable_hours_predicted']/stats['total_hours']*100:.1f}%)\n")
                else:
                    f.write("(0.0%)\n")
                
                f.write(f"  Non-Billable Hours (Predicted): {stats['non_billable_hours_predicted']:.2f} ")
                if stats['total_hours'] > 0:
                    f.write(f"({stats['non_billable_hours_predicted']/stats['total_hours']*100:.1f}%)\n")
                else:
                    f.write("(0.0%)\n")
                    
                f.write(f"  Billable Hours (Actual): {stats['billable_hours_actual']:.2f} ")
                if stats['total_hours'] > 0:
                    f.write(f"({stats['billable_hours_actual']/stats['total_hours']*100:.1f}%)\n")
                else:
                    f.write("(0.0%)\n")
                
                f.write(f"  Non-Billable Hours (Actual): {stats['non_billable_hours_actual']:.2f} ")
                if stats['total_hours'] > 0:
                    f.write(f"({stats['non_billable_hours_actual']/stats['total_hours']*100:.1f}%)\n")
                else:
                    f.write("(0.0%)\n")
        
        print(f"Summary report generated successfully: {output_path}")
        return output_path
        
    except Exception as e:
        raise Exception(f"Error generating summary report: {str(e)}")

if __name__ == "__main__":
    # Simple test with mock data
    mock_data = [
        {
            "employee_name": "John Doe",
            "project_name": "Client XYZ Implementation",
            "customer_name": "XYZ Corp",
            "organization_name": "Solitwork A/S",
            "hours": 8,
            "description": "Implemented new features for the client dashboard",
            "date": "2025-02-15",
            "is_billable_predicted": True,
            "is_billable_actual": True,
            "is_approved": True,
            "classification_confidence": 0.95,
            "classification_reasoning": "Direct client implementation work"
        },
        {
            "employee_name": "Jane Smith",
            "project_name": "Internal Tools",
            "customer_name": "Solitwork A/S",
            "organization_name": "Solitwork A/S",
            "hours": 2,
            "description": "Weekly team meeting",
            "date": "2025-02-15",
            "is_billable_predicted": False,
            "is_billable_actual": False,
            "is_approved": False,
            "classification_confidence": 0.87,
            "classification_reasoning": "Internal meeting not related to client work"
        },
        {
            "employee_name": "Alice Johnson",
            "project_name": "Client ABC Support",
            "customer_name": "ABC Inc",
            "organization_name": "Solitwork A/S",
            "hours": 4,
            "description": "Support call with client",
            "date": "2025-02-16",
            "is_billable_predicted": True,
            "is_billable_actual": False,
            "is_approved": False,
            "classification_confidence": 0.75,
            "classification_reasoning": "Client support work is typically billable"
        }
    ]
    
    try:
        csv_path = generate_csv_output(mock_data)
        metrics, confusion_matrix = evaluate_classification_performance(mock_data)
        print("Performance metrics:")
        print(metrics)
        print("\nConfusion matrix:")
        print(confusion_matrix)
        summary_path = generate_summary_report(mock_data)
        print(f"Files generated: {csv_path}, {summary_path}")
    except Exception as e:
        print(f"Error: {str(e)}") 