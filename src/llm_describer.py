import os
import time
import json
from typing import Dict, Any, List, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv

class DescriptionError(Exception):
    """Exception raised when description generation fails"""
    pass

def initialize_client() -> AzureOpenAI:
    """Initialize the Azure OpenAI client"""
    load_dotenv()
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not endpoint or not api_key:
        raise ValueError("Azure OpenAI endpoint and API key must be set in environment variables")
    
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key
    )
    
    return client

def format_project_prompt(project_data: Dict[str, Any]) -> str:
    """Format the project data into a prompt for description generation"""
    
    # Extract employees (limited to first 3 characters for each employee)
    employees = [emp[:3] for emp in project_data.get('employees', [])]
    employees_str = ", ".join(employees) if employees else "No employees listed"
    
    # Extract tasks with their hours and billable/non-billable distribution
    tasks = project_data.get('tasks', {})
    tasks_str = ""
    for task_name, task_info in tasks.items():
        billable_pct = (task_info.get('billable_hours', 0) / task_info.get('total_hours', 1)) * 100 if task_info.get('total_hours', 0) > 0 else 0
        non_billable_pct = 100 - billable_pct
        potential_billable_pct = (task_info.get('potential_billable_hours', 0) / task_info.get('total_hours', 1)) * 100 if task_info.get('total_hours', 0) > 0 else 0
        tasks_str += f"\n- {task_name}: {task_info.get('total_hours', 0):.2f} hours ({billable_pct:.1f}% billable, {non_billable_pct:.1f}% non-billable, {potential_billable_pct:.1f}% potential missed billing)"
    
    if not tasks_str:
        tasks_str = "No tasks listed"
    
    # Calculate total project statistics
    total_hours = project_data.get('total_hours', 0)
    billable_hours = project_data.get('billable_hours', 0)
    non_billable_hours = project_data.get('non_billable_hours', 0)
    potential_billable_hours = project_data.get('potential_billable_hours', 0)
    
    billable_pct = (billable_hours / total_hours) * 100 if total_hours > 0 else 0
    non_billable_pct = 100 - billable_pct
    potential_billable_pct = (potential_billable_hours / total_hours) * 100 if total_hours > 0 else 0
    
    approved_pct = project_data.get('approved_percentage', 0)
    not_approved_pct = project_data.get('not_approved_percentage', 0)
    
    return f"""
    Please generate a short and concise project performance summary based on the following data:
    
    Project Name: {project_data.get('project_name', 'Unknown Project')}
    Key Employees: {employees_str}
    
    Tasks and Hours:{tasks_str}
    
    Total Hours: {total_hours:.2f}
    Billable/Non-billable Distribution: {billable_pct:.1f}% billable, {non_billable_pct:.1f}% non-billable
    Potential Missed Billing: {potential_billable_hours:.2f} hours ({potential_billable_pct:.1f}% of total)
    Approval Status: {approved_pct:.1f}% approved, {not_approved_pct:.1f}% not approved
    """

def generate_project_description(project_data: Dict[str, Any], prompts_list: List[Dict[str, Any]], max_retries: int = 3) -> str:
    """
    Generate a project summary description using Azure OpenAI
    
    Args:
        project_data: Dictionary containing project statistics
        prompts_list: List to collect prompt data
        max_retries: Maximum number of retries for API calls
    
    Returns:
        Generated project description as a string
    """
    client = initialize_client()
    model_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    # Format the project data for the prompt
    prompt = format_project_prompt(project_data)
    
    # Add prompt to the list along with project information
    prompt_data = {
        "project_name": project_data.get('project_name', 'Unknown Project'),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt
    }
    prompts_list.append(prompt_data)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a professional project reporting assistant for a IT consulting/SaaS company.
Your task is to generate concise, insightful project performance summaries based on timesheet data and a predicted approval for each registered hour.
The registered hour is disapproved if it's registered as non-billable but contains indicators it should be billable, such as ticket numbers or customer-specific work.
Don't use adjectives to describe the performance.

Guidelines:
- Keep the summary short (3-5 sentences maximum)
- Highlight key insights about potential missed billing opportunities
- Focus on non-billable hours that should have been billable
- Mention approval rates only if they are noteworthy (very high or very low)
- Use professional, business-appropriate language
- Focus on the most important metrics and tasks
- Do not speculate beyond the data provided
"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=250,
                temperature=0.3,
                model=model_deployment
            )
            
            project_description = response.choices[0].message.content.strip()
            
            # Add the response to the prompt data
            prompt_data["response"] = project_description
            
            return project_description
                
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff
                time.sleep(2 ** attempt)
                continue
            else:
                error_message = f"Description generation failed after {max_retries} attempts: {str(e)}"
                prompt_data["error"] = error_message
                raise DescriptionError(error_message)
    
    # This should not be reached due to the exception in the final retry
    error_message = "Description generation failed with an unknown error"
    prompt_data["error"] = error_message
    raise DescriptionError(error_message)

def prepare_project_data(classified_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Prepare project data from classified entries
    
    Args:
        classified_data: List of dictionaries with classified hours data
        
    Returns:
        Dictionary of project data organized by project name
    """
    projects = {}
    
    for entry in classified_data:
        project_name = entry.get('project_name', 'Unknown Project')
        employee_name = entry.get('employee_name', 'Unknown Employee')
        task_name = entry.get('task_name', 'Unknown Task')
        hours = float(entry.get('hours', 0))
        
        # Use both is_billable_key and project_is_billable to determine if hours are billable
        # is_billable_key = 1 when the registered hour is billable
        # project_is_billable = 'Yes' when the project is billable
        is_billable_key = entry.get('is_billable_key') == 1
        project_is_billable = entry.get('project_is_billable') == 'Yes'
        
        # A billable hour must have is_billable_key=1 (individual hour is billable)
        is_billable = is_billable_key
        
        # Check if this is a non-billable hour that should be billable
        # Focus on finding non-billable hours that should be billable
        description = str(entry.get('description', '')).lower()
        has_ticket_number = '#' in description
        customer_name = str(entry.get('customer_name', '')).lower()
        potential_billable = (not is_billable and project_is_billable == 'Yes') or \
                            (not is_billable and has_ticket_number) or \
                            (not is_billable and customer_name and 'internal' not in project_name.lower())
        
        is_approved = entry.get('is_approved_predicted') == True
        
        # Initialize project if not exists
        if project_name not in projects:
            projects[project_name] = {
                'project_name': project_name,
                'employees': set(),
                'tasks': {},
                'total_hours': 0,
                'billable_hours': 0,
                'non_billable_hours': 0,
                'potential_billable_hours': 0,
                'approved_hours': 0,
                'not_approved_hours': 0
            }
        
        # Add employee
        projects[project_name]['employees'].add(employee_name)
        
        # Initialize task if not exists
        if task_name not in projects[project_name]['tasks']:
            projects[project_name]['tasks'][task_name] = {
                'total_hours': 0,
                'billable_hours': 0,
                'non_billable_hours': 0,
                'potential_billable_hours': 0,
                'approved_hours': 0,
                'not_approved_hours': 0
            }
        
        # Update task stats
        projects[project_name]['tasks'][task_name]['total_hours'] += hours
        if is_billable:
            projects[project_name]['tasks'][task_name]['billable_hours'] += hours
        else:
            projects[project_name]['tasks'][task_name]['non_billable_hours'] += hours
            
        if potential_billable:
            projects[project_name]['tasks'][task_name]['potential_billable_hours'] += hours
            
        if is_approved:
            projects[project_name]['tasks'][task_name]['approved_hours'] += hours
        else:
            projects[project_name]['tasks'][task_name]['not_approved_hours'] += hours
        
        # Update project totals
        projects[project_name]['total_hours'] += hours
        if is_billable:
            projects[project_name]['billable_hours'] += hours
        else:
            projects[project_name]['non_billable_hours'] += hours
            
        if potential_billable:
            projects[project_name]['potential_billable_hours'] += hours
            
        if is_approved:
            projects[project_name]['approved_hours'] += hours
        else:
            projects[project_name]['not_approved_hours'] += hours
    
    # Calculate percentages and convert employee sets to lists
    for project_name, project in projects.items():
        total_hours = project['total_hours']
        
        # Calculate approval percentages
        if total_hours > 0:
            project['approved_percentage'] = (project['approved_hours'] / total_hours) * 100
            project['not_approved_percentage'] = (project['not_approved_hours'] / total_hours) * 100
            project['potential_billable_percentage'] = (project['potential_billable_hours'] / total_hours) * 100 if project['non_billable_hours'] > 0 else 0
        else:
            project['approved_percentage'] = 0
            project['not_approved_percentage'] = 0
            project['potential_billable_percentage'] = 0
        
        # Convert employee set to sorted list
        project['employees'] = sorted(list(project['employees']))
    
    return projects

def save_prompts_to_json(prompts_list: List[Dict[str, Any]], output_dir: str = "output", summarizer_type: str = "project") -> str:
    """
    Save the collected prompts to a JSON file
    
    Args:
        prompts_list: List of dictionaries containing prompt data
        output_dir: Directory to save the JSON file
        summarizer_type: Type of summarizer ('project' or 'customer')
        
    Returns:
        Path to the generated JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate file path with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{summarizer_type}_prompts_{timestamp}.json")
    
    # Save prompts to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompts_list, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(prompts_list)} {summarizer_type} prompts to {output_path}")
    return output_path

def generate_project_descriptions(classified_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Generate descriptions for all projects in the classified data
    
    Args:
        classified_data: List of dictionaries with classified hours data
        
    Returns:
        Dictionary of project data with descriptions
    """
    # Prepare data for each project
    projects = prepare_project_data(classified_data)
    
    # Create a list to collect all prompts
    prompts_list = []
    
    # Generate descriptions for each project
    for project_name, project_data in projects.items():
        try:
            description = generate_project_description(project_data, prompts_list)
            project_data['description'] = description
        except DescriptionError as e:
            print(f"Error generating description for project {project_name}: {str(e)}")
            project_data['description'] = f"Project summary unavailable: {str(e)}"
    
    # Save all collected prompts to a JSON file
    save_prompts_to_json(prompts_list, summarizer_type="project")
    
    return projects

def format_customer_prompt(customer_data: Dict[str, Any]) -> str:
    """Format the customer data into a prompt for description generation"""
    
    # Extract projects with their hours and billable/non-billable distribution
    projects = customer_data.get('projects', {})
    projects_str = ""
    for project_name, project_info in projects.items():
        billable_pct = (project_info.get('billable_hours', 0) / project_info.get('total_hours', 1)) * 100 if project_info.get('total_hours', 0) > 0 else 0
        non_billable_pct = 100 - billable_pct
        potential_billable_pct = (project_info.get('potential_billable_hours', 0) / project_info.get('total_hours', 1)) * 100 if project_info.get('total_hours', 0) > 0 else 0
        projects_str += f"\n- {project_name}: {project_info.get('total_hours', 0):.2f} hours ({billable_pct:.1f}% billable, {non_billable_pct:.1f}% non-billable, {potential_billable_pct:.1f}% potential missed billing)"
    
    if not projects_str:
        projects_str = "No projects listed"
    
    # Calculate total customer statistics
    total_hours = customer_data.get('total_hours', 0)
    billable_hours = customer_data.get('billable_hours', 0)
    non_billable_hours = customer_data.get('non_billable_hours', 0)
    potential_billable_hours = customer_data.get('potential_billable_hours', 0)
    
    billable_pct = (billable_hours / total_hours) * 100 if total_hours > 0 else 0
    non_billable_pct = 100 - billable_pct
    potential_billable_pct = (potential_billable_hours / total_hours) * 100 if total_hours > 0 else 0
    
    approved_pct = customer_data.get('approved_percentage', 0)
    not_approved_pct = customer_data.get('not_approved_percentage', 0)
    
    return f"""
    Please generate a short and concise customer performance summary based on the following data:
    
    Customer Name: {customer_data.get('customer_name', 'Unknown Customer')}
    
    Projects and Hours:{projects_str}
    
    Total Hours: {total_hours:.2f}
    Billable/Non-billable Distribution: {billable_pct:.1f}% billable, {non_billable_pct:.1f}% non-billable
    Potential Missed Billing: {potential_billable_hours:.2f} hours ({potential_billable_pct:.1f}% of total)
    Approval Status: {approved_pct:.1f}% approved, {not_approved_pct:.1f}% not approved
    """

def generate_customer_description(customer_data: Dict[str, Any], prompts_list: List[Dict[str, Any]], max_retries: int = 3) -> str:
    """
    Generate a customer summary description using Azure OpenAI
    
    Args:
        customer_data: Dictionary containing customer statistics
        prompts_list: List to collect prompt data
        max_retries: Maximum number of retries for API calls
    
    Returns:
        Generated customer description as a string
    """
    client = initialize_client()
    model_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    # Format the customer data for the prompt
    prompt = format_customer_prompt(customer_data)
    
    # Add prompt to the list along with customer information
    prompt_data = {
        "customer_name": customer_data.get('customer_name', 'Unknown Customer'),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt
    }
    prompts_list.append(prompt_data)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a professional customer reporting assistant for a IT consulting/SaaS company.
Your task is to generate concise, insightful customer performance summaries based on timesheet data and billing analytics.
The focus should be on identifying potential missed billing opportunities, especially where non-billable hours appear to have billable characteristics.
Don't use adjectives to describe the performance.

Guidelines:
- Keep the summary short (3-5 sentences maximum)
- Highlight key insights about potential missed billing opportunities
- Focus on non-billable hours that should have been billable
- Mention any suspicious patterns of non-billable hours for external customers
- Use professional, business-appropriate language
- Do not speculate beyond the data provided
"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=250,
                temperature=0.3,
                model=model_deployment
            )
            
            customer_description = response.choices[0].message.content.strip()
            
            # Add the response to the prompt data
            prompt_data["response"] = customer_description
            
            return customer_description
                
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff
                time.sleep(2 ** attempt)
                continue
            else:
                error_message = f"Customer description generation failed after {max_retries} attempts: {str(e)}"
                prompt_data["error"] = error_message
                raise DescriptionError(error_message)
    
    # This should not be reached due to the exception in the final retry
    error_message = "Customer description generation failed with an unknown error"
    prompt_data["error"] = error_message
    raise DescriptionError(error_message)

def prepare_customer_data(classified_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Prepare customer data from classified entries
    
    Args:
        classified_data: List of dictionaries with classified hours data
        
    Returns:
        Dictionary of customer data organized by customer name
    """
    customers = {}
    
    for entry in classified_data:
        customer_name = entry.get('customer_name', 'Unknown Customer')
        project_name = entry.get('project_name', 'Unknown Project')
        employee_name = entry.get('employee_name', 'Unknown Employee')
        task_name = entry.get('task_name', 'Unknown Task')
        hours = float(entry.get('hours', 0))
        
        # Use is_billable_key to determine if hours are billable
        is_billable_key = entry.get('is_billable_key') == 1
        project_is_billable = entry.get('project_is_billable') == 'Yes'
        
        is_billable = is_billable_key
        
        # Check if this is a non-billable hour that should be billable
        description = str(entry.get('description', '')).lower()
        has_ticket_number = '#' in description
        potential_billable = (not is_billable and project_is_billable == 'Yes') or \
                            (not is_billable and has_ticket_number) or \
                            (not is_billable and customer_name and 'internal' not in project_name.lower())
        
        is_approved = entry.get('is_approved_predicted') == True
        
        # Initialize customer if not exists
        if customer_name not in customers:
            customers[customer_name] = {
                'customer_name': customer_name,
                'projects': {},
                'employees': set(),
                'total_hours': 0,
                'billable_hours': 0,
                'non_billable_hours': 0,
                'potential_billable_hours': 0,
                'approved_hours': 0,
                'not_approved_hours': 0
            }
        
        # Add employee
        customers[customer_name]['employees'].add(employee_name)
        
        # Initialize project if not exists
        if project_name not in customers[customer_name]['projects']:
            customers[customer_name]['projects'][project_name] = {
                'total_hours': 0,
                'billable_hours': 0,
                'non_billable_hours': 0,
                'potential_billable_hours': 0,
                'approved_hours': 0,
                'not_approved_hours': 0
            }
        
        # Update project stats
        customers[customer_name]['projects'][project_name]['total_hours'] += hours
        if is_billable:
            customers[customer_name]['projects'][project_name]['billable_hours'] += hours
        else:
            customers[customer_name]['projects'][project_name]['non_billable_hours'] += hours
            
        if potential_billable:
            customers[customer_name]['projects'][project_name]['potential_billable_hours'] += hours
            
        if is_approved:
            customers[customer_name]['projects'][project_name]['approved_hours'] += hours
        else:
            customers[customer_name]['projects'][project_name]['not_approved_hours'] += hours
        
        # Update customer totals
        customers[customer_name]['total_hours'] += hours
        if is_billable:
            customers[customer_name]['billable_hours'] += hours
        else:
            customers[customer_name]['non_billable_hours'] += hours
        
        if potential_billable:
            customers[customer_name]['potential_billable_hours'] += hours
            
        if is_approved:
            customers[customer_name]['approved_hours'] += hours
        else:
            customers[customer_name]['not_approved_hours'] += hours
    
    # Calculate percentages and convert sets to sorted lists
    for customer_name, customer in customers.items():
        total_hours = customer['total_hours']
        
        # Calculate approval percentages
        if total_hours > 0:
            customer['approved_percentage'] = (customer['approved_hours'] / total_hours) * 100
            customer['not_approved_percentage'] = (customer['not_approved_hours'] / total_hours) * 100
            customer['potential_billable_percentage'] = (customer['potential_billable_hours'] / total_hours) * 100 if customer['non_billable_hours'] > 0 else 0
        else:
            customer['approved_percentage'] = 0
            customer['not_approved_percentage'] = 0
            customer['potential_billable_percentage'] = 0
        
        # Convert sets to sorted lists
        customer['employees'] = sorted(list(customer['employees']))
    
    return customers

def generate_customer_descriptions(classified_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Generate descriptions for all customers in the classified data
    
    Args:
        classified_data: List of dictionaries with classified hours data
        
    Returns:
        Dictionary of customer data with descriptions
    """
    # Prepare data for each customer
    customers = prepare_customer_data(classified_data)
    
    # Create a list to collect all prompts
    prompts_list = []
    
    # Generate descriptions for each customer
    for customer_name, customer_data in customers.items():
        try:
            description = generate_customer_description(customer_data, prompts_list)
            customer_data['description'] = description
        except DescriptionError as e:
            print(f"Error generating description for customer {customer_name}: {str(e)}")
            customer_data['description'] = f"Customer summary unavailable: {str(e)}"
    
    # Save all collected prompts to a JSON file
    save_prompts_to_json(prompts_list, output_dir="output", summarizer_type="customer")
    
    return customers 