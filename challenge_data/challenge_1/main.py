import json
import re
import requests
from sklearn.metrics import accuracy_score, f1_score

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

def evaluate(ground_truth_file, submission_file, phase_code, leaderboard_threshold=0.9, **kwargs):
    """
    Evaluate the participant's submission.
    
    Args:
        submission_file (str): Path to the participant's submission JSON file.
        ground_truth_file (str): Path to the ground truth JSON file.
        phase_code (str): The phase code of the evaluation ("dev" or "test").
        leaderboard_threshold (float): Threshold to determine if results are in the top leaderboard.
        **kwargs: Additional arguments for communication, such as a Discord webhook.
        
    Returns:
        dict: Evaluation metrics including accuracy, F1-score, and link validation results.
    """
    # Optional: Notify via Discord if webhook is provided
    discord_webhook = kwargs.get("discord_webhook")

    output = {}

    # Load the ground truth
    with open(ground_truth_file, "r") as gt_file:
        ground_truth_data = json.load(gt_file)

    # Load the submission
    with open(submission_file, "r") as sub_file:
        submission_data = json.load(sub_file)

    # Validate URLs
    model_link_valid = 1 if validate_url(submission_data.get("HF_Link", ""), "Model Link") else 0
    pdf_link_valid = 1 if validate_url(submission_data.get("PDF_Link", ""), "PDF Report") else 0

    if model_link_valid == 0 or pdf_link_valid == 0:
        #raise ValueError("Submission contains invalid Model Link or PDF Report URL.")
        print("Submission contains invalid Model Link or PDF Report URL.")

    # Parse predictions and true labels
    ground_truth = {item["video_id"]: item["true_label"] for item in ground_truth_data["predictions"]}
    submission = {item["video_id"]: item["prediction"] for item in submission_data["predictions"]}

    # Ensure submission and ground truth match
    if set(submission.keys()) != set(ground_truth.keys()):
        raise ValueError("Submission and ground truth files must have the same video keys.")
        #print("Submission and ground truth files must have the same video keys.")

    # Extract predictions and true labels
    y_true = [ground_truth[video_id] for video_id in ground_truth]
    y_pred = [submission[video_id] for video_id in submission]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Check if the submission qualifies for the top leaderboard
    if accuracy >= leaderboard_threshold and discord_webhook:
        try:
            requests.post(discord_webhook, json={
                "content": f"Submission in phase '{phase_code}' achieved top leaderboard status with accuracy: {accuracy:.2f}."
            })
        except requests.RequestException as e:
            print(f"Error sending notification to Discord: {e}")
    
    output["result"] = [
            {
                "train_split": {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "model_link_valid": model_link_valid,
                    "pdf_link_valid": pdf_link_valid
                }
            }
        ]
    output["submission_result"] = output["result"][0]["train_split"]
    print("Completed evaluation for Dev Phase")
    print(output)

    return output

    # Return metrics
    #return {
    #    "accuracy": accuracy,
    #    "f1_score": f1,
    #    "model_link_valid": model_link_valid,
    #    "pdf_link_valid": pdf_link_valid
    #}

# Example usage (for debugging purposes)
#if __name__ == "__main__":
#    submission_path = "submission.json"  # Replace with actual submission file path
#    ground_truth_path = "annotations/test_annotations_testsplit.json"  # Replace with actual ground truth file path
#    phase = "test"  # or "test" "dev"
#    discord_hook = "https://discord.com/api/webhooks/1319704452093186118/pU1pn_lH-A1vOu5IxlHmCJn-Zy9PybyH_ra5dWclQAJcrai1vaHuEYhGI9W0EG7dmbn-"  # Replace with actual webhook URL
#    results = evaluate(submission_path, ground_truth_path, phase, leaderboard_threshold=0.9, discord_webhook=discord_hook)
#    print(results)
