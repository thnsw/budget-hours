#!/usr/bin/env python3
import os
import sys
import pandas as pd
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime

def upload_data(classified_data: List[Dict[str, Any]], mode: str = 'upload', output_dir: str = 'output'):
    """
    Uploads classified time entry data to SQL Server or saves it as a CSV file.

    Args:
        classified_data: A list of dictionaries, where each dictionary represents a time entry 
                         record including original fields and classification results 
                         (e.g., 'is_approved_predicted', 'classification_confidence', 
                         'classification_reasoning').
        mode: The operation mode. 'upload' to upload to SQL Server, 'test' to save as CSV.
        output_dir: The directory to save the CSV file in 'test' mode.
    """
    # Load environment variables
    load_dotenv()
    
    # Convert the input list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(classified_data)
    
    # Define target columns and rename if needed
    column_mapping = {
        'fact_dw_id': 'Fact_DW_ID',
        'date': 'Date',
        'employee_name': 'EmployeeName',
        'project_name': 'ProjectName',
        'customer_name': 'CustomerName',
        'task_name': 'TaskName',
        'hours': 'Hours',
        'project_is_billable': 'ProjectIsBillable',
        'is_billable_key': 'IsBillable',
        'description': 'Description',
        'is_approved_key': 'IsApproved',
        'is_approved_predicted': 'IsApprovedPredicted',
        'classification_confidence': 'ClassificationConfidence',
        'classification_reasoning': 'ClassificationReasoning',
        'PromptSystem': 'PromptSystem',
        'PromptUser': 'PromptUser',
        'ModelName': 'ModelName',
        'PredictionTimestamp': 'PredictionTimestamp',
        'PredictionLatencyMS': 'PredictionLatencyMS',
        'TokenCountPrompt': 'TokenCountInput',
        'TokenCountCompletion': 'TokenCountOutput',
        'Environment': 'Environment'
    }
    
    # Rename columns to match target SQL table
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Select only the columns we want to upload
    target_columns = [
        'Fact_DW_ID', 'Date', 'EmployeeName', 'ProjectName', 'CustomerName', 'TaskName', 
        'Hours', 'ProjectIsBillable', 'IsBillable', 'Description',
        'IsApproved', 'IsApprovedPredicted', 'ClassificationConfidence', 
        'ClassificationReasoning',
        'PromptSystem',
        'PromptUser',
        'ModelName',
        'PredictionTimestamp',
        'PredictionLatencyMS',
        'TokenCountInput',
        'TokenCountOutput',
        'Environment'
    ]
    
    # Create a new DataFrame with only the columns that exist
    # This prevents errors if some columns from the mapping are missing
    available_columns = [col for col in target_columns if col in df.columns]
    final_df = df[available_columns].copy()
    
    # Handle the 'test' mode
    if mode.lower() == 'test':
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Replace newlines for CSV compatibility
        df_copy_for_csv = final_df.copy()
        if 'PromptSystem' in df_copy_for_csv.columns:
            df_copy_for_csv['PromptSystem'] = df_copy_for_csv['PromptSystem'].str.replace('\n', ' ', regex=False)
        if 'PromptUser' in df_copy_for_csv.columns:
            df_copy_for_csv['PromptUser'] = df_copy_for_csv['PromptUser'].str.replace('\n', ' ', regex=False)
        
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"registered_hours_test_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Save DataFrame to CSV (using the modified copy)
        df_copy_for_csv.to_csv(csv_path, index=False)
        print(f"Test data saved to CSV: {csv_path}")
        return csv_path
    
    # Handle the 'upload' mode
    elif mode.lower() == 'upload':
        try:
            # Get database connection details from environment variables
            server = os.getenv('SQL_SERVER')
            database = os.getenv('SQL_DATABASE')
            username = os.getenv('SQL_USER')
            password = os.getenv('SQL_PASSWORD')
            
            # Check if connection string is provided directly
            connection_string = os.getenv('SQL_CONNECTION_STRING')
            
            if connection_string:
                print("Connecting to SQL Server using connection string...")
                engine = create_engine(connection_string, fast_executemany=True)
            else:
                # Verify all required parameters are available
                if not all([server, database, username, password]):
                    raise ValueError("Missing database connection parameters. Check your .env file.")
                
                print("Connecting to SQL Server...")
                conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
                engine = create_engine(conn_str, fast_executemany=True)
            
            # Define the target schema and table name
            schema_name = 'AIPowerBIData'
            table_name = 'RegisteredHours'
            
            print(f"Uploading data to {schema_name}.{table_name}...")
            final_df.to_sql(
                name=table_name,
                schema=schema_name,
                con=engine,
                if_exists='append',
                index=False
            )
            print(f"Successfully uploaded {len(final_df)} records to the database.")
            return True
            
        except Exception as e:
            print(f"Error uploading data to SQL Server: {str(e)}")
            return False
    
    else:
        print(f"Invalid mode: {mode}. Use 'upload' or 'test'.")
        return False

if __name__ == "__main__":
    print("This module is designed to be imported and used by other scripts.")
    print("Example usage:")
    print("    from upload import upload_data")
    print("    # Upload to SQL Server")
    print("    upload_data(classified_data, mode='upload')")
    print("    # Save to CSV for testing")
    print("    upload_data(classified_data, mode='test', output_dir='test_outputs')") 