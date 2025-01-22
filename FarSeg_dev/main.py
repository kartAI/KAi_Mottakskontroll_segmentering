# FarSeg_dev/main.py

# Libraries:

from Functionality import generalFunctions as gf
from Functionality.dataStatistics import statistics
from Program.mainInference import mainInference
from Program.mainTrain import mainTrain
from Program.mainValidation import mainValidation

# Main program:

print("""
Hello!

Here you got four alternatives:

1. Train a new FarSeg segmentation model
2. Use a trained FarSeg model to perform segmentation
3. Validate results from inference with a FarSeg model
4. See relevant statistics about your geographic data

Just write a number between 1-4 to choose action.
""")

choice = gf.get_valid_input("Perform (1-4): ", gf.positiveNumber)

if choice.isdigit():
    choice = int(choice)
    if choice == 1:
        if __name__ == '__main__':
            mainTrain()
    elif choice == 2:
        mainInference()
    elif choice == 3:
        mainValidation()
    elif choice == 4:
        geopackages = gf.get_valid_input("Where are the stored geopackages(?): ", gf.doesPathExists)
        stat = statistics(geopackages)
        stat.main()

print("Thanks, and good bye!")
