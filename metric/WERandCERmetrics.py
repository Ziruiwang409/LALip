import re


def extract_words(sentence):
    words = re.findall(r'\b\w+\b', sentence)
    return words


def CalculateErrorRate(ground_truth, prediction, method='WER'):
    assert method.upper() in ['WER', 'CER'], f"method: {method} is not supported"

    # extract words from ground truth and prediction
    ground_truth = extract_words(ground_truth)
    prediction = extract_words(prediction)
    if method.upper() == 'CER':
        ground_truth = ''.join(ground_truth)
        prediction = ''.join(prediction)

    # Create a tabular to store the distances
    distances = [[0] * (len(prediction) + 1) for _ in range(len(ground_truth) + 1)]

    # Initialize the matrix
    for i in range(len(ground_truth) + 1):
        distances[i][0] = i
    for j in range(len(prediction) + 1):
        distances[0][j] = j

    # Populate the matrix with edit distances
    for i in range(1, len(ground_truth) + 1):
        for j in range(1, len(prediction) + 1):
            if ground_truth[i - 1] == prediction[j - 1]:
                cost = 0
            else:
                cost = 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,  # Deletion
                distances[i][j - 1] + 1,  # Insertion
                distances[i - 1][j - 1] + cost,  # Substitution
            )

    # Calculate CER
    error_rate = distances[len(ground_truth)][len(prediction)] / len(ground_truth)
    return error_rate


if __name__ == '__main__':
    ground_truth = "the quick brown fox jumped over the lazy dog"
    prediction = "the brown fox jumped over the lazy dog"

    error_rate = CalculateErrorRate(ground_truth, prediction, method='WER')

    print(error_rate)

