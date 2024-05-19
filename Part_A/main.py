# SALTYKOV DANIIL 07/05/2024
# F1 RACING RESULTS
#
# Copyright: GNU Public License http://www.gnu.org/licenses/
# 20220372@student.act.edu

from tabulate import tabulate
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import common.print_instructions as pi

# Initializing the global data frame for data storage
data = pd.DataFrame()


def menu_option_check(value, min_option, max_option):
    if ascii(value) < ascii(min_option) or ascii(value) > ascii(max_option):
        print(f"Invalid input! Please make sure your option choice is between {min_option} and {max_option}.")
        return False
    return True  # Thing to discuss


def display_data():
    print("-----------------------------------------------------------------")
    input("Press Enter to continue...")


def read_data():
    print("-----------------------------------------------------------------")
    global data
    data = pd.read_csv('res/partA_input_data.csv', sep='\t')
    # display(data)
    print(tabulate(data, headers='keys'))
    print(data.head())
    input("Press Enter to continue...")
#     TODO: Add more appealing data printing


def lap_search():
    # global data
    print("-----------------------------------------------------------------")
    limit = int(input("Enter the limit of laps to search by: "))
    print(data[data.LAPS >= limit].sort_values('GRAND PRIX'))
    input("Press Enter to continue...")


def avg_lap_time():
    print("-----------------------------------------------------------------")
    input("Press Enter to continue...")


def field_sort():
    print("-----------------------------------------------------------------")
    input("Press Enter to continue...")


def column_graph():
    print("-----------------------------------------------------------------")
    input("Press Enter to continue...")


def main():
    print("=================================================================")
    print("F1 GRAND PRIX RACING DATA & STATISTICS FOR THE 2023 RACING SEASON")
    print("=================================================================")

    while True:
        print("-----------------------------------------------------------------")
        pi.display_menu_instructions()
        option = input("Enter your choice: ")

        match option:
            case '0':
                pi.print_legend()
                input("Press Enter to continue...")
            case '1':
                read_data()
            case '2':
                lap_search()
            case '3':
                avg_lap_time()
            case '4':
                field_sort()
            case '5':
                column_graph()
            case '6':
                print("-----------------------------------------------------------------")
                print("You selected option 6.\nIt's time to say goodbye then...\nBye!")
                break
            case default:
                print("Invalid choice. Please try again!")
                input("Press Enter to continue...")


if __name__ == "__main__":
    main()
