# SALTYKOV DANIIL 27/05/2024
# MLP Classification
#
#
#
# Copyright: GNU Public License http://www.gnu.org/licenses/
# 20220372@student.act.edu

# Import of required libraries
import os
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import common.print_instructions as pi
import re


# The main function
def main():
    # TODO(change the header)
    print("=====================================================================")
    print("USER MENU: MLP CLASSIFICATION OF THE ###### DATA SET (UCI REPOSITORY)")
    print("=====================================================================")

    while True:
        print("--------------------------------------------------------------------------------------------------")
        pi.mlp_menu_instructions()
        option = input("Enter your choice: ").replace(" ", "")

        # Checks if the user use option 1 first to read the data
        if option != '1' and option != '6':
            print("\nNot enough data to proceed! Please run \"1\" option to load the data first.")
            input("Press Enter to continue...")
        else:
            # Match case for menu implementation
            match option:
                # Additional case for more detailed information about menu options
                case '0':
                    pi.mlp_legend()
                    input("Press Enter to continue...")
                case '1':
                    print()
                case '2':
                    print()
                case '3':
                    print()
                case '4':
                    print()
                case '5':
                    print()
                case '6':
                    print("-------------------------------------------------------------------------------------------"
                          "-------")
                    print("You selected option 6.\nIt's time to say goodbye then...\nBye!")
                    quit(0)
                # The default case for wrong data input
                case _:
                    print("Invalid choice. Please try again!")
                    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
