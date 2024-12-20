
import csv
import re
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def validate_url(url, url_type):
    """
    Validate a URL by checking its format and attempting a HEAD request.
    
    Args:
        url (str): The URL to validate.
        url_type (str): Type of URL (e.g., 'Model Link', 'PDF Report').

    Returns:
        bool: True if the URL is valid, otherwise False.
    """
    # Basic URL format validation
    regex = re.compile(
        r'^(https?:\/\/)?'  # http:// or https://
        r'(([a-zA-Z0-9_-]+\.)+[a-zA-Z]{2,})'  # Domain name
        r'(\/[-a-zA-Z0-9@:%_+.~#?&/=]*)?$'  # Path
    )
    if not re.match(regex, url):
        print(f"Invalid {url_type} URL format: {url}")
        return False

    # Attempting a HEAD request to check URL reachability
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        if response.status_code == 200:
            return True
        else:
            print(f"{url_type} URL not reachable (Status code: {response.status_code}): {url}")
            return False
    except requests.RequestException as e:
        print(f"Error validating {url_type} URL: {url}. Error: {e}")
        return False

def evaluate(submission_file, ground_truth_file):
    """
    Evaluate the participant's submission.
    
    Args:
        submission_file (str): Path to the participant's submission CSV file.
        ground_truth_file (str): Path to the ground truth CSV file.
        
    Returns:
        dict: Evaluation metrics including accuracy, precision, recall, F1-score, confusion matrix, and link validation metrics.
    """
    # Read the ground truth
    with open(ground_truth_file, "r") as gt_file:
        ground_truth_reader = csv.reader(gt_file)
        ground_truth = {row[0]: row[1] for idx, row in enumerate(ground_truth_reader) if idx >= 2}

    # Read the submission
    with open(submission_file, "r") as sub_file:
        submission_reader = csv.reader(sub_file)
        submission_rows = list(submission_reader)

    # Validate URLs in the first two rows
    model_link = submission_rows[0][0].split(":")[-1].strip()
    pdf_link = submission_rows[1][0].split(":")[-1].strip()

    model_link_valid = 1 if validate_url(model_link, "Model Link") else 0
    pdf_link_valid = 1 if validate_url(pdf_link, "PDF Report") else 0

    if model_link_valid == 0 or pdf_link_valid == 0:
        raise ValueError("Submission contains invalid Model Link or PDF Report URL.")

    # Parse predictions from the submission
    submission = {row[0]: row[1] for idx, row in enumerate(submission_rows) if idx >= 2}

    # Ensure submission and ground truth match
    if set(submission.keys()) != set(ground_truth.keys()):
        raise ValueError("Submission and ground truth files must have the same video keys.")

    # Extract predictions and true labels
    y_true = [ground_truth[video_id] for video_id in ground_truth]
    y_pred = [submission[video_id] for video_id in submission]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    confusion = confusion_matrix(y_true, y_pred)

    # Return metrics
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion.tolist(),  # Convert to list for JSON compatibility
        "model_link_valid": model_link_valid,
        "pdf_link_valid": pdf_link_valid
    }

# Example usage (for debugging purposes)
if __name__ == "__main__":
    submission_path = "submission.csv"  # Replace with actual submission file path
    ground_truth_path = "ground_truth.csv"  # Replace with actual ground truth file path
    results = evaluate(submission_path, ground_truth_path)
    print(results)
