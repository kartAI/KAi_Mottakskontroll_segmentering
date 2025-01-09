# FarSeg_div/Functionality/generalFunctions.py

# Libraries:

# Functions

def validInput(ans, input):
    """
    Checks if the user answer is valid input

    Args:
        ans (string): input answer from user
        input (list of strings): valid input
    Returns:
        A boolean value
    """
    ans = ans.lower()
    if ans in input:
        return True
    else:
        return False

def yesNo(ans):
    """
    Checks if an answer is either yes or no
    
    Args:
        ans (string): input answer from user
    Returns:
        A boolean value
    """
    ans = ans.lower()
    if ans.lower() == "y":
        return True
    elif ans == "n":
        return False
