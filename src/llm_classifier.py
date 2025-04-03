import os
import json
import time
from typing import Dict, Any, Union, List, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

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
        Dictionary with original data plus classification results
    """
    client = initialize_client()
    model_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    # Format the hour entry for the prompt
    prompt = format_classification_prompt(hour_entry)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are a billable hours classifier. Your task is to determine if the hours 
                        logged should be classified as 'billable' or 'non-billable'. 
                        
                        Return your response as a valid JSON object with the following structure:
                        {
                            "classification": "billable" or "non-billable",
                            "confidence": a number between 0 and 1,
                            "reasoning": "brief explanation of your decision"
                        }
                        
                        Focus on:
                        - Task description and nature of work
                        - Project type (client projects are typically billable)
                        - Internal activities are typically non-billable
                        - Administrative work is typically non-billable
                        - Customer name - external customers typically have billable work
                        - Organization context - may indicate internal vs external work
                        """
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.1,
                model=model_deployment
            )
            
            content = response.choices[0].message.content
            
            # Try to parse the JSON response
            try:
                classification_result = extract_json_from_response(content)
                
                # Validate expected fields
                if not all(k in classification_result for k in ["classification", "confidence", "reasoning"]):
                    raise json.JSONDecodeError("Missing required fields", content, 0)
                
                # Add the classification to the original entry
                result = hour_entry.copy()
                result.update({
                    "is_billable_predicted": classification_result["classification"] == "billable",
                    "classification_confidence": classification_result["confidence"],
                    "classification_reasoning": classification_result["reasoning"]
                })
                
                # Preserve the actual IsBillableKey value for comparison if it exists
                if "_is_billable_key" in hour_entry:
                    result["is_billable_actual"] = hour_entry["_is_billable_key"] == 1
                
                # Preserve the actual IsApprovedKey value for evaluation if it exists
                if "_is_approved_key" in hour_entry:
                    result["is_approved"] = hour_entry["_is_approved_key"] == 1
                
                return result
                
            except (json.JSONDecodeError, KeyError) as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise ClassificationError(f"Failed to parse classification after {max_retries} attempts: {str(e)}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff
                time.sleep(2 ** attempt)
                continue
            else:
                raise ClassificationError(f"Classification failed after {max_retries} attempts: {str(e)}")
    
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
    Please classify the following hours entry as billable or non-billable:
    
    Employee: {hour_entry.get('employee_name')}
    Project: {hour_entry.get('project_name')}
    Customer: {hour_entry.get('customer_name')}
    Organization: {hour_entry.get('organization_name')}
    Hours: {hour_entry.get('hours')}
    Description: {hour_entry.get('description')}
    Date: {hour_entry.get('date')}
    """

def classify_batch(hours_data: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
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
            
            # Simple rate limiting - pause after each batch
            if (i + 1) % batch_size == 0 and i < len(hours_data) - 1:
                print(f"Processed {i + 1}/{len(hours_data)} entries. Pausing for rate limiting...")
                time.sleep(2)
                
        except ClassificationError as e:
            print(f"Error classifying entry {i}: {str(e)}")
            # Add the entry with an error flag
            error_entry = entry.copy()
            error_entry.update({
                "is_billable_predicted": None,
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