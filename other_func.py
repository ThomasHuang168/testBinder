# This Python script is for other functions used in
# the jupyter notebook for the experiments involving different
# types of datasets.

def find_gt_in_log(log_file_path):
    r"""Function to find ground truth in log file of a model.
    
    Arguments:
        log_file_path {str} -- file path of the log file
    
    Returns:
        ground_truth {float} -- Ground Truth stored in a model's log file
    """
    ground_truth_str = ''
    with open(log_file_path, "r") as file:
        for line in file:
            if ('ground_truth' in line):
                ground_truth_str = line

    i = 0
    while not (ground_truth_str[i].isdigit()):
        i += 1
    ground_truth = float(ground_truth_str[i:])
    return ground_truth

# Function to find top limit for later graph plotting
def find_top_limit(gt):
    if (gt == 0):
        top_limit = 0.1
    else:
        top_limit = 1.1*gt
    return top_limit

# Function to find threshold for drawing vertical line
# in graph indicating ground truth has reached threshold
def find_threshold(thres, gt):
    return thres*gt
