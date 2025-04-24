import os
import json
import time
from typing import Dict, Any, Union, List, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from datetime import datetime

class ClassificationError(Exception):
    """Exception raised when classification fails"""
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

def classify_hours(hour_entry: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """
    Classify if hours are billable or non-billable using Azure OpenAI
    
    Args:
        hour_entry: Dictionary containing hours data
        max_retries: Maximum number of retries for malformed JSON responses
    
    Returns:
        Dictionary with original data plus classification results and metadata
    """
    client = initialize_client()
    model_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    environment = os.getenv('APP_ENVIRONMENT', 'development')
    
    # Define prompts
    system_prompt = """You are an hour approval classifier for SolitWork, a IT consulting/Saas company that delivers financial, ESG, and analytics services.
Your task is to predict approval probability for registered hours.

Return your response as a valid JSON object with the following structure:
{
    "approval_probability": a float number between 0.0 and 1.0,
    "reasoning": "One sentence explaning your decision based on the input data."
}

Primary focus:
- Hours containing ticket numbers (#) in the description and registered as non-billable (Is Billable: 0) should be disapproved (low approval probability), as they indicate potential billing opportunities
- Hours containing customer names  in the description and registered as non-billable (Is Billable: 0) should be disapproved (low approval probability), as they indicate potential billing opportunities
- External hours spent on customer work but registered as non-billable should be disapproved if the description does not explain why it is non-billable
Secondary focus:
- If the description is comprehensive it suggests billable work and should indicate higher approval probability.

Example 1 Input:
Employee: Alice
Project: Internal
Task: Internal
Hours: 2
Project Is Billable: No
Is Billable: 0
Description: #1234 Fixed data load issue.

Example 1 Output:
{
    "approval_probability": 0.1,
    "reasoning": "Description contains ticket number and is therefore related to customer work."
}

Example 2 Input:
Employee: Bob
Project: Client ABC
Task: Meeting
Hours: 8
Project Is Billable: Yes
Is Billable: 0
Description: Meeting with TDC.

Example 2 Output:
{
    "approval_probability": 0.1,
    "reasoning": "Contains ticket number and client meeting but registered as non-billable, should likely be billable."
}

Example 3 Input:
Employee: Charlie
Project: Client DEF Migration
Task: Data Validation
Hours: 6
Project Is Billable: Yes
Is Billable: 1
Description: Helped customer with issue #45678.

Example 3 Output:
{
    "approval_probability": 1.0,
    "reasoning": "Customer work with ticket number marked as non-billable without explanation why it shouldn't be billed."
}
"""
    user_prompt = format_classification_prompt(hour_entry)

    start_time = time.perf_counter()
    prediction_timestamp = datetime.utcnow()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                max_tokens=200,
                temperature=0.1,
                model=model_deployment,
                response_format={"type": "json_object"}
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            content = response.choices[0].message.content
            usage_info = response.usage

            # Try to parse the JSON response
            try:
                classification_result = extract_json_from_response(content)
                
                # Validate expected fields
                if not all(k in classification_result for k in ["approval_probability", "reasoning"]):
                    raise json.JSONDecodeError("Missing required fields 'approval_probability' or 'reasoning'", content, 0)
                if not isinstance(classification_result["approval_probability"], (int, float)) or not (0 <= classification_result["approval_probability"] <= 1):
                     raise ValueError("Invalid value for 'approval_probability'. Must be a float between 0 and 1.")

                # Add the classification and metadata to the original entry
                result = hour_entry.copy()
                approval_probability = float(classification_result["approval_probability"])
                result.update({
                    "is_approved_predicted": approval_probability > 0.5,
                    "classification_confidence": approval_probability,
                    "classification_reasoning": classification_result["reasoning"],
                    "PromptSystem": system_prompt,
                    "PromptUser": user_prompt,
                    "ModelName": model_deployment,
                    "PredictionTimestamp": prediction_timestamp,
                    "PredictionLatencyMS": latency_ms,
                    "TokenCountPrompt": usage_info.prompt_tokens if usage_info else None,
                    "TokenCountCompletion": usage_info.completion_tokens if usage_info else None,
                    "Environment": environment
                })
                
                # Preserve the actual IsApprovedKey value for evaluation if it exists
                if "_is_approved_key" in hour_entry:
                    result["is_approved"] = hour_entry["_is_approved_key"] == 1
                
                return result
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ClassificationError(f"Failed to parse or validate classification after {max_retries} attempts: {str(e)}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff
                time.sleep(2 ** attempt)
                continue
            else:
                raise ClassificationError(f"Classification API call failed after {max_retries} attempts: {str(e)}")
    
    # This should not be reached due to the exception in the final retry
    raise ClassificationError("Classification failed with an unknown error")

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract JSON from the response text, handling cases where the model might
    include explanatory text before or after the JSON
    """
    try:
        # First try to parse the entire response as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If that fails, raise the original error
                raise
        else:
            raise json.JSONDecodeError("No JSON object found in response", response_text, 0)

def format_classification_prompt(hour_entry: Dict[str, Any]) -> str:
    """Format the hour entry data into a prompt for classification"""
    # Only include fields relevant for classification
    return f"""
    Please classify the following hours entry as approved or not approved:
    
    Employee: {hour_entry.get('employee_name')}
    Project: {hour_entry.get('project_name')}
    Customer: {hour_entry.get('customer_name')}
    Task: {hour_entry.get('task_name')}
    Hours: {hour_entry.get('hours')}
    Project Is Billable: {hour_entry.get('project_is_billable')}
    Registered hour is billable: {hour_entry.get('is_billable_key')}
    Billable Amount: {hour_entry.get('billable_amount')}
    Description: {hour_entry.get('description')}
    """

def classify_batch(hours_data: List[Dict[str, Any]], batch_size: int = 100) -> List[Dict[str, Any]]:
    """
    Classify a batch of hours data with rate limiting
    
    Args:
        hours_data: List of hour entries to classify
        batch_size: Number of entries to process before pausing
        
    Returns:
        List of classified hour entries
    """
    results = []
    
    for i, entry in enumerate(hours_data):
        try:
            # Remove the fields that should not be used for classification
            classification_entry = {k: v for k, v in entry.items() if not k.startswith('_')}
            
            classified_entry = classify_hours(classification_entry)
            
            # Add back the metadata fields
            for k, v in entry.items():
                if k.startswith('_'):
                    classified_entry[k] = v
            
            results.append(classified_entry)
            
            # Simple batch progress logging without rate limiting
            # gpt-4o-mini can handle 12K requests per minute (Default tier) or 300K requests per minute (Enterprise tier)
            if (i + 1) % batch_size == 0 and i < len(hours_data) - 1:
                print(f"Processed {i + 1}/{len(hours_data)} entries.")
                
        except ClassificationError as e:
            print(f"Error classifying entry {i}: {str(e)}")
            # Add the entry with an error flag
            error_entry = entry.copy()
            error_entry.update({
                "is_approved_predicted": None,
                "classification_confidence": 0,
                "classification_reasoning": f"Error: {str(e)}",
                "classification_error": True
            })
            results.append(error_entry)
    
    return results

if __name__ == "__main__":
    # Simple test with a mock entry
    mock_entry = {
        "employee_name": "John Doe",
        "project_name": "Client XYZ Implementation",
        "customer_name": "XYZ Corp",
        "organization_name": "Solitwork A/S",
        "hours": 8,
        "description": "Implemented new features for the client dashboard",
        "date": "2025-02-15"
    }
    
    try:
        result = classify_hours(mock_entry)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}") 