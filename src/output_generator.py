import os
import csv
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from fpdf import FPDF
from fpdf.fonts import FontFace
from PIL import Image

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

class PDFReportGenerator:
    def __init__(self, output_dir="output"):
        """Initialize the PDF generator with an output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Paths to theme images
        self.header_image_path = os.path.join("src", "sw-theme", "Header-uden-background.png")
        self.footer_image_path = os.path.join("src", "sw-theme", "Footer-uden-background.png")
        
        # Pre-calculate image heights for positioning
        try:
            header_img = Image.open(self.header_image_path)
            self.header_height = (header_img.height / header_img.width) * 210  # Maintain aspect ratio
            
            footer_img = Image.open(self.footer_image_path)
            self.footer_height = (footer_img.height / footer_img.width) * 210  # Maintain aspect ratio
        except Exception as e:
            print(f"Warning: Could not load theme images: {str(e)}")
            self.header_height = 20  # Default fallback values
            self.footer_height = 20
    
    def header_function(self, pdf):
        """Add header to each page"""
        try:
            pdf.image(self.header_image_path, x=0, y=0, w=210)  # Full page width
        except Exception as e:
            print(f"Warning: Could not add header image: {str(e)}")
    
    def footer_function(self, pdf):
        """Add footer to each page"""
        try:
            pdf.image(self.footer_image_path, x=0, y=297-self.footer_height, w=210)
        except Exception as e:
            print(f"Warning: Could not add footer image: {str(e)}")
    
    def create_pdf_report(self, classified_data: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Create a PDF report with overall statistics and project details
        
        Args:
            classified_data: List of dictionaries with classified hours data
            output_path: Optional path for the output file. If None, a default path will be generated.
            
        Returns:
            Path to the generated PDF file
        """
        if not classified_data:
            raise ValueError("No data provided for PDF report generation")
        
        # Generate a default output path if none provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.output_dir}/summary_report_{timestamp}.pdf"
        
        # Initialize PDF (A4 format by default: 210x297 mm)
        pdf = FPDF()
        
        # Set header and footer functions
        pdf.header = lambda: self.header_function(pdf)
        pdf.footer = lambda: self.footer_function(pdf)
        
        # Adjust margins to accommodate header and footer
        pdf.set_margins(10, self.header_height + 10, 10)
        pdf.set_auto_page_break(auto=True, margin=self.footer_height + 10)
        
        # Calculate overall statistics (removed call to evaluate_classification_performance)
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
        project_employees = {}
        
        for entry in classified_data:
            project = entry.get('project_name', 'Unknown')
            employee = entry.get('employee_name', 'Unknown')
            
            # Initialize project stats if not exists
            if project not in projects:
                projects[project] = {
                    'total_hours': 0,
                    'approved_hours_predicted': 0,
                    'not_approved_hours_predicted': 0,
                    'entries': 0
                }
            
            # Initialize project-employee tracking
            if project not in project_employees:
                project_employees[project] = {}
                
            if employee not in project_employees[project]:
                project_employees[project][employee] = {
                    'total_hours': 0,
                    'approved_hours_predicted': 0,
                    'not_approved_hours_predicted': 0,
                    'entries': 0
                }
            
            # Update project stats
            projects[project]['entries'] += 1
            hours = float(entry.get('hours', 0))
            projects[project]['total_hours'] += hours
            
            if entry.get('is_approved_predicted') is True:
                projects[project]['approved_hours_predicted'] += hours
            elif entry.get('is_approved_predicted') is False:
                projects[project]['not_approved_hours_predicted'] += hours
            
            # Update employee stats within project
            project_employees[project][employee]['entries'] += 1
            project_employees[project][employee]['total_hours'] += hours
            
            if entry.get('is_approved_predicted') is True:
                project_employees[project][employee]['approved_hours_predicted'] += hours
            elif entry.get('is_approved_predicted') is False:
                project_employees[project][employee]['not_approved_hours_predicted'] += hours
        
        # Create cover page
        pdf.add_page()
        pdf.set_font("Times", "B", 24)
        
        # Position title in middle of page
        y_position = (297 - self.header_height - self.footer_height) / 2 - 20
        pdf.set_y(self.header_height + y_position)
        pdf.cell(0, 10, "Hours Approval Summary Report", ln=True, align="C")
        
        # Add overall statistics section (removed Classification Performance Metrics section)
        pdf.add_page()
        pdf.set_font("Times", "B", 16)
        pdf.cell(0, 10, "Overall Statistics", ln=True)
        pdf.set_font("Times", "", 10)
        pdf.ln(5)
        
        # Define column widths for statistics table
        stats_col_widths = [80, 50, 50]
        
        # Define table styles - moved outside conditional blocks to avoid reference errors
        col_widths = [100, 40]
        heading_style = FontFace(emphasis="BOLD", color=0, fill_color=(200, 200, 200))
        light_grey = (240, 240, 240)
        
        data = [
            ["Metric", "Value", "Percentage"],
            ["Total Entries Processed", str(total_entries), "100%"],
            ["Approved Entries", str(approved_predicted), f"{approved_predicted/total_entries*100:.1f}%"],
            ["Not Approved Entries", str(not_approved_predicted), f"{not_approved_predicted/total_entries*100:.1f}%"],
            ["Error Entries (Prediction Failed)", str(error_entries), f"{error_entries/total_entries*100:.1f}%"],
            ["Total Hours Processed", f"{total_hours:.2f}", "100%"],
            ["Approved Hours", f"{approved_hours_predicted:.2f}", f"{approved_hours_predicted/total_hours*100:.1f}%"],
            ["Not Approved Hours", f"{not_approved_hours_predicted:.2f}", f"{not_approved_hours_predicted/total_hours*100:.1f}%"]
        ]
        
        with pdf.table(
            col_widths=stats_col_widths,
            text_align=["LEFT", "CENTER", "CENTER"],
            line_height=6,
            headings_style=heading_style,
            cell_fill_mode="ROWS",
            cell_fill_color=light_grey
        ) as table:
            for i, row_data in enumerate(data):
                row = table.row()
                for cell_data in row_data:
                    row.cell(cell_data)
        
        # Add statistics by project section
        pdf.add_page()
        pdf.set_font("Times", "B", 16)
        pdf.cell(0, 10, "Statistics by Project", ln=True)
        
        # Define column widths for project statistics tables
        project_stats_col_widths = [80, 50, 50]
        employee_stats_col_widths = [60, 20, 30, 40, 40]
        
        # Add a section for each project
        for project, stats in projects.items():
            pdf.ln(5)
            pdf.set_font("Times", "B", 14)
            pdf.cell(0, 10, f"Project: {project}", ln=True)
            pdf.set_font("Times", "", 10)
            
            # Project statistics table
            data = [
                ["Metric", "Value", "Percentage"],
                ["Total Entries", str(stats['entries']), "100%"],
                ["Total Hours", f"{stats['total_hours']:.2f}", "100%"],
                ["Approved Hours", f"{stats['approved_hours_predicted']:.2f}", 
                 f"{stats['approved_hours_predicted']/stats['total_hours']*100:.1f}%" if stats['total_hours'] > 0 else "0%"],
                ["Not Approved Hours", f"{stats['not_approved_hours_predicted']:.2f}", 
                 f"{stats['not_approved_hours_predicted']/stats['total_hours']*100:.1f}%" if stats['total_hours'] > 0 else "0%"]
            ]
            
            with pdf.table(
                col_widths=project_stats_col_widths,
                text_align=["LEFT", "CENTER", "CENTER"],
                line_height=6,
                headings_style=heading_style,
                cell_fill_mode="ROWS",
                cell_fill_color=light_grey
            ) as table:
                for i, row_data in enumerate(data):
                    row = table.row()
                    for cell_data in row_data:
                        row.cell(cell_data)
            
            # Employee statistics within project
            if project in project_employees and project_employees[project]:
                pdf.ln(5)
                pdf.set_font("Times", "B", 12)
                pdf.cell(0, 10, "Employee Statistics", ln=True)
                pdf.set_font("Times", "", 10)
                
                # Create table headers
                employee_data = [["Employee", "Entries", "Total Hours", "Approved Hours", "Not Approved Hours"]]
                
                # Add data for each employee
                for employee, emp_stats in project_employees[project].items():
                    employee_data.append([
                        employee,
                        str(emp_stats['entries']),
                        f"{emp_stats['total_hours']:.2f}",
                        f"{emp_stats['approved_hours_predicted']:.2f}",
                        f"{emp_stats['not_approved_hours_predicted']:.2f}"
                    ])
                
                with pdf.table(
                    col_widths=employee_stats_col_widths,
                    text_align=["LEFT", "CENTER", "CENTER", "CENTER", "CENTER"],
                    line_height=6,
                    headings_style=heading_style,
                    cell_fill_mode="ROWS",
                    cell_fill_color=light_grey
                ) as table:
                    for i, row_data in enumerate(employee_data):
                        row = table.row()
                        for cell_data in row_data:
                            row.cell(cell_data)
            
            # Add disapproved hours details table
            disapproved_entries = [entry for entry in classified_data 
                                  if entry.get('project_name') == project 
                                  and entry.get('is_approved_predicted') is False]
            
            if disapproved_entries:
                pdf.ln(10)
                pdf.set_font("Times", "B", 12)
                pdf.cell(0, 10, "Disapproved Hours Detail", ln=True)
                pdf.set_font("Times", "", 10)
                
                # Create table headers for disapproved hours
                disapproved_col_widths = [25, 25, 15, 40, 85]
                disapproved_data = [["Employee", "Date", "Hours", "Description", "Explanation"]]
                
                # Add data for each disapproved entry
                for entry in disapproved_entries:
                    # Limit explanation length to fit in table cell
                    explanation = entry.get('classification_reasoning', '')
                    if len(explanation) > 80:
                        explanation = explanation[:77] + "..."
                    
                    # Limit description length to fit in table cell
                    description = entry.get('description', '')
                    if len(description) > 35:
                        description = description[:32] + "..."
                        
                    disapproved_data.append([
                        entry.get('employee_name', 'Unknown'),
                        str(entry.get('date', 'N/A')),
                        f"{float(entry.get('hours', 0)):.2f}",
                        description,
                        explanation
                    ])
                    print(f"disapproved_data: {disapproved_data}")
                
                with pdf.table(
                    col_widths=disapproved_col_widths,
                    text_align=["LEFT", "LEFT", "CENTER", "LEFT", "LEFT"],
                    line_height=7,
                    headings_style=heading_style,
                    cell_fill_mode="ROWS",
                    cell_fill_color=light_grey
                ) as table:
                    for i, row_data in enumerate(disapproved_data):
                        row = table.row()
                        for cell_data in row_data:
                            row.cell(cell_data)
            
            # Add a page break between projects (except for the last one)
            if list(projects.keys()).index(project) < len(projects) - 1:
                pdf.add_page()
        
        # Save the PDF file
        pdf.output(output_path)
        print(f"PDF report generated successfully: {output_path}")
        return output_path

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
    
    # Create a PDF report using the PDFReportGenerator
    try:
        pdf_generator = PDFReportGenerator()
        
        # Generate the output path if none is provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/summary_report_{timestamp}.pdf"
        
        # Create and save the PDF report
        pdf_path = pdf_generator.create_pdf_report(classified_data, output_path)
        return pdf_path
        
    except Exception as e:
        raise Exception(f"Error generating PDF summary report: {str(e)}")

if __name__ == "__main__":
    # Simple test with mock data
    mock_data = [
        {
            "employee_name": "John Doe", "project_name": "Project A", "customer_name": "Cust A", 
            "hours": 5, "description": "Work", "date": "2024-01-01",
            "is_approved_predicted": True, "is_approved": True, 
            "classification_confidence": 0.9, "classification_reasoning": "Clear and specific description that aligns well with the project requirements."
        },
        {
            "employee_name": "Jane Smith", "project_name": "Project B", "customer_name": "Cust B", 
            "hours": 3, "description": "Stuff", "date": "2024-01-02",
            "is_approved_predicted": False, "is_approved": True, 
            "classification_confidence": 0.7, "classification_reasoning": "Description is too vague. 'Stuff' does not provide sufficient information about the work performed on this billable project."
        },
        {
            "employee_name": "John Doe", "project_name": "Project A", "customer_name": "Cust A", 
            "hours": 2, "description": "More work", "date": "2024-01-03",
            "is_approved_predicted": True, "is_approved": False, 
            "classification_confidence": 0.8, "classification_reasoning": "Description matches project tasks, although more detail would be helpful for future submissions."
        },
        {
            "employee_name": "Peter Jones", "project_name": "Project B", "customer_name": "Cust B", 
            "hours": 6, "description": "Meeting", "date": "2024-01-04",
            "is_approved_predicted": False, "is_approved": False, 
            "classification_confidence": 0.95, "classification_reasoning": "6 hours for simply 'Meeting' is excessive without further details. Client projects require specific descriptions of meeting content and outcomes."
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
            
    except Exception as e:
        print(f"Error in test: {str(e)}") 