import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime, timedelta

def create_connection():
    """Create a connection to the SQL Server database using environment variables"""
    load_dotenv()
    
    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DATABASE")
    username = os.getenv("SQL_USERNAME")
    password = os.getenv("SQL_PASSWORD")
    
    connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server"
    
    try:
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        raise ConnectionError(f"Failed to connect to database: {str(e)}")

def extract_hours_data(limit: int = None, debug: bool = True) -> List[Dict[Any, Any]]:
    """
    Extract hours data from the SQL Server for February 2025
    
    Args:
        limit: Optional limit on number of records to extract
        debug: Whether to print debug information
    
    Returns:
        List of dictionaries containing hours data
    """
    engine = create_connection()
    
    # Calculate the date 7 days ago
    seven_days_ago = datetime.now() - timedelta(days=7)
    start_date_str = seven_days_ago.strftime('%Y%m%d')

    # Query to extract latest updated registered hour using DW_ID and newest DW_Batch_Created. 
    # Hours is greater than 0 to avoid latest negation post.
    query = """
	WITH LatestBillableEntries AS (
		SELECT
			f.Fact_DW_ID,
			f.Hours,
			f.Date,
			p.ProjectName,
			t.TaskName,
			p.IsBillable AS ProjectIsBillable,
			c.CustomerName,
			d.DescriptionID AS Description,
			f.IsBillableKey,
			e.EmployeeName,
			f.DW_ID,
			f.DW_Batch_Created,
			ROW_NUMBER() OVER (PARTITION BY f.DW_ID ORDER BY f.DW_Batch_Created DESC) AS RowNum
		FROM
			[PowerBIData].[vPowerBiData_Harvest_Harvest_data_All] f
		LEFT JOIN
			[PowerBIData].[DimProject_Tabular_Flat] p ON f.ProjectKey = p.ProjectKey
		LEFT JOIN
			[PowerBIData].[DimCustomer_Tabular_Flat] c ON f.CustomerKey = c.CustomerKey
		LEFT JOIN
			[PowerBIData].[DimOrganization_Tabular_Flat] o ON f.OrganizationKey = o.OrganizationKey
		LEFT JOIN
			[PowerBIData].[DimDescription] d ON f.DescriptionKey = d.DescriptionKey
		LEFT JOIN
			[PowerBIData].[DimTask_Tabular_Flat] t on f.TaskKey = t.TaskKey
		LEFT JOIN
			[PowerBIData].[DimPeriod] period ON f.Period = period.Period
		LEFT JOIN
			(SELECT DISTINCT EmployeeKey, EmployeeName FROM [PowerBIData].[DimEmployee_Tabular_Flat]) e ON f.EmployeeKey = e.EmployeeKey
		WHERE
			Hours > 0
	)
	SELECT * FROM LatestBillableEntries
	WHERE
		RowNum = 1
		AND Date >= :start_date
		AND (CustomerName like '%PINDSTRUP MOSEBRUG A/S%' OR CustomerName like 'SW CST 3' OR CustomerName like 'SØSTRENE GRENES IMPORT A/S')
		AND TaskName not in ('Barsel', 'Sygdom // Sickness')
		AND (EmployeeName like 'THN - Thomas Nissen' OR EmployeeName like 'PRO - Peter Rokkjær' OR EmployeeName like 'MGU - Morten Gunnersen')
		AND IsBillableKey = 0
    """
    
    if limit:
        query += f" ORDER BY DW_ID DESC OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
    else:
        query += " ORDER BY DW_ID DESC"
    
    try:
        with engine.connect() as connection:
            # Pass start_date as a parameter
            result = connection.execute(text(query), {"start_date": start_date_str})
            data = [dict(row._mapping) for row in result]
            print(f"SQL query returned {len(data)} rows for dates >= {start_date_str}")
            return data
    except Exception as e:
        raise Exception(f"Error extracting data: {str(e)}")
    finally:
        engine.dispose()

def extract_data_for_classification(limit: int = None) -> List[Dict[Any, Any]]:
    """
    Extract only the data needed for LLM classification
    
    Args:
        limit: Optional limit on number of records to extract
    
    Returns:
        List of dictionaries containing data formatted for LLM classification
    """
    raw_data = extract_hours_data(limit=limit)
    
    # Format data for LLM classification - only include fields needed for classification
    classification_data = []
    for entry in raw_data:
        # Make sure entry is a dictionary before accessing with get()
        if not isinstance(entry, dict):
            print(f"Warning: Expected dictionary but got {type(entry)}: {entry}")
            continue
            
        # Clean description to remove page breaks that can break CSV output
        description = entry.get("Description", "")
        if description:
            # Replace newlines, carriage returns, and other problematic characters
            description = str(description).replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
            
        formatted_entry = {
            "fact_dw_id": entry.get("Fact_DW_ID"),
            "employee_name": entry.get("EmployeeName"),
            "project_name": entry.get("ProjectName"),
            "customer_name": entry.get("CustomerName"),
            "hours": entry.get("Hours"),
            "description": description,
            "project_is_billable": entry.get("ProjectIsBillable"),
            "task_name": entry.get("TaskName"),
            "date": entry.get("Date"),
            # Include all original fields to ensure no data is lost
            "is_billable_key": entry.get("IsBillableKey"),
        }
        classification_data.append(formatted_entry)
    
    return classification_data

if __name__ == "__main__":
    # Simple test to verify extraction works
    try:
        data = extract_data_for_classification()
        print(f"Successfully extracted {len(data)} records")
        if data:
            print("Sample record:")
            print(pd.DataFrame([data[0]]).to_string())
    except Exception as e:
        print(f"Error: {str(e)}") 