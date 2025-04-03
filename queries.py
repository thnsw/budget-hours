"""
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
        ROW_NUMBER() OVER (PARTITION BY f.Date, f.DW_ID, e.EmployeeName, t.TaskName ORDER BY f.DW_Batch_Created DESC) AS RowNum
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
        period.YearName = '2024' AND 
        period.MonthName = 'Dec' AND 
        o.OrganizationID = 'SWDK' AND 
        p.IsBillable = 'Yes' AND
        --e.EmployeeName = 'JET - Jan Ettema' AND 
        --p.ProjectName = 'Support Finance' AND 
        --f.Date = '20241204' AND
        f.BillableAmount > 0 AND
		f.IsApprovedKey = 0
		
)
SELECT * FROM LatestBillableEntries WHERE RowNum = 1
order by Date desc
"""


"""
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
		f.*
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
        f.DW_ID = 2551802983
	ORDER BY
		f.Date desc, f.DW_Batch_Created desc, f.Fact_DW_ID desc
"""