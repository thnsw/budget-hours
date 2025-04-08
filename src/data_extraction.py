import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from typing import List, Dict, Any

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
    
    # Query to extract latest updated registered hour using DW_ID and newest DW_Batch_Created. 
    # Hours is greater than 0 to avoid latest negation post.
    query = """
	WITH LatestBillableEntries AS (
		SELECT
			f.Hours,
			f.Date,
			p.ProjectID,
			p.ProjectName,
			t.TaskName,
			p.IsBillable AS ProjectIsBillable,
			c.CustomerName,
			d.DescriptionID AS Description,
			f.BillableAmount,
			f.BillableRate,
			f.IsBillableKey,
			f.IsApprovedKey,
			e.EmployeeName,
			f.DW_ID,
			f.DW_Batch_Created,
			ROW_NUMBER() OVER (PARTITION BY f.DW_ID ORDER BY f.DW_Batch_Created DESC) AS RowNum
		FROM
			[PowerBIData].[vPowerBiData_Harvest_Harvest_data_All] f
		JOIN
			[PowerBIData].[DimProject_Tabular_Flat] p ON f.ProjectKey = p.ProjectKey
		JOIN
			[PowerBIData].[DimCustomer_Tabular_Flat] c ON f.CustomerKey = c.CustomerKey
		JOIN
			[PowerBIData].[DimOrganization_Tabular_Flat] o ON f.OrganizationKey = o.OrganizationKey
		JOIN
			[PowerBIData].[DimDescription] d ON f.DescriptionKey = d.DescriptionKey
		JOIN
			[PowerBIData].[DimTask_Tabular_Flat] t on f.TaskKey = t.TaskKey
		JOIN
			[PowerBIData].[DimPeriod] period ON f.Period = period.Period
		LEFT JOIN
			(SELECT DISTINCT EmployeeKey, EmployeeName FROM [PowerBIData].[DimEmployee_Tabular_Flat]) e ON f.EmployeeKey = e.EmployeeKey
		WHERE
			Hours > 0
	)
	SELECT * FROM LatestBillableEntries
	WHERE
		RowNum = 1
		AND Date like '20250305'
		--AND ProjectName = 'Conscia Support'
		AND (CustomerName like '%PINDSTRUP MOSEBRUG A/S%' OR CustomerName like 'SW CST 3' OR CustomerName like 'SØSTRENE GRENES IMPORT A/S')
		AND TaskName not in ('Barsel', 'Sygdom // Sickness')
		--AND CustomerName like '%cst%'
		--AND Hours > 0
		--AND IsBillableKey = 1
		--AND IsApprovedKey = 0
		--AND DW_ID = 2617911362
		AND Description like '%#%'
    """
    
    if limit:
        query += f" ORDER BY DW_ID DESC OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
    else:
        query += " ORDER BY DW_ID DESC"
    
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            data = [dict(row._mapping) for row in result]
            print(f"SQL query returned {len(data)} rows")
            return data
    except Exception as e:
        raise Exception(f"Error extracting data: {str(e)}")
    finally:
        engine.dispose()

def extract_data_for_classification() -> List[Dict[Any, Any]]:
    """
    Extract only the data needed for LLM classification
    
    Returns:
        List of dictionaries containing data formatted for LLM classification
    """
    raw_data = extract_hours_data()
    
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
            "employee_name": entry.get("EmployeeName"),
            "project_name": entry.get("ProjectName"),
            "customer_name": entry.get("CustomerName"),
            "hours": entry.get("Hours"),
            "description": description,
            "project_is_billable": entry.get("ProjectIsBillable"),
            "task_name": entry.get("TaskName"),
            "billable_amount": entry.get("BillableAmount"),
            "date": entry.get("Date"),
            # Include all original fields to ensure no data is lost
            "project_id": entry.get("ProjectID"),
            "dw_id": entry.get("DW_ID"),
            "is_billable_key": entry.get("IsBillableKey"),
            # Store actual value for later comparison but don't include in LLM input
            "is_approved_key": entry.get("IsApprovedKey")
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