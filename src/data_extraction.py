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

def extract_hours_data(limit: int = None) -> List[Dict[Any, Any]]:
    """
    Extract hours data from the SQL Server for February 2025
    
    Args:
        limit: Optional limit on number of records to extract
    
    Returns:
        List of dictionaries containing hours data
    """
    engine = create_connection()
    
    # Query to extract data for February 2025 with all relevant information
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
            f.DW_Batch_Created,
            f.DW_ID,
            ROW_NUMBER() OVER (PARTITION BY f.DW_ID, f.Date, e.EmployeeName, t.TaskName ORDER BY f.DW_Batch_Created DESC) AS RowNum
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
            period.YearName = '2024' 
            AND period.MonthName = 'Dec' 
            AND o.OrganizationID = 'SWDK'
            AND p.IsBillable = 'Yes' 
            AND e.EmployeeName = 'JET - Jan Ettema' 
            --AND p.ProjectName = 'Support Finance' 
            AND f.Date = '20241204'
            AND f.BillableAmount > 0
            -- AND f.IsApprovedKey = 0		
    )
    SELECT * FROM LatestBillableEntries WHERE RowNum = 1
    order by Date desc
    """
    
    if limit:
        query += f" ORDER BY f.Date OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
    
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            data = [dict(row._mapping) for row in result]
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
            raise TypeError(f"Expected dictionary but got {type(entry)}: {entry}")
            
        formatted_entry = {
            "employee_name": entry.get("EmployeeName"),
            "project_name": entry.get("ProjectName"),
            "customer_name": entry.get("CustomerName"),
            "organization_name": entry.get("OrganizationName"),
            "hours": entry.get("Hours"),
            "description": entry.get("Description"),
            "date": entry.get("Date").strftime("%Y-%m-%d") if hasattr(entry.get("Date"), "strftime") else entry.get("Date"),
            "project_is_billable": entry.get("ProjectIsBillable"),
            # Store actual values for later comparison but don't include in LLM input
            "_is_billable_key": entry.get("IsBillableKey"),
            "_is_approved_key": entry.get("IsApprovedKey")
        }
        classification_data.append(formatted_entry)
    
    return classification_data

if __name__ == "__main__":
    # Simple test to verify extraction works
    try:
        data = extract_data_for_classification()
        print(f"Successfully extracted {len(data)} records for February 2025")
        if data:
            print("Sample record:")
            print(pd.DataFrame([data[0]]).to_string())
    except Exception as e:
        print(f"Error: {str(e)}") 