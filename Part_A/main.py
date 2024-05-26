# SALTYKOV DANIIL 07/05/2024
# F1 RACING RESULTS
#
#
#
# Copyright: GNU Public License http://www.gnu.org/licenses/
# 20220372@student.act.edu

from random import *
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import common.print_instructions as pi
import re

# Initializing the global data frame for data storage
data = pd.DataFrame()


def input_error_checker(message, min_option, max_option):
    while True:
        value = input(message).replace(" ", "")
        if min_option <= value <= max_option:
            return int(value)
        else:
            print(f"\nInvalid input! Please make sure your option choice is between {min_option} and {max_option}.")


def display_data(data_input) -> None:
    if data_input.empty:
        print("\nThere is no data matching your request.")
    else:
        print(tabulate(data_input, headers='keys'))


# Convert time in format mm:ss.ms to a double.
def time_to_double(time_str):
    minutes, seconds_ds = time_str.split(':')
    seconds, deciseconds = seconds_ds.split('.')

    # Calculates the final time in seconds and return the double variable
    total_seconds = int(minutes) * 60 + int(seconds) + int(deciseconds) / 10
    return total_seconds


# Convert a double to time in format mm:ss.ms.
def double_to_time(time_double):
    minutes = int(time_double // 60)
    seconds = int(time_double % 60)
    deciseconds = round((time_double - minutes * 60 - seconds) * 10)

    # In case of rounding deciseconds up to 10, increase seconds by one integer
    if deciseconds == 10:
        deciseconds = 0
        seconds += 1

    # Aggregates the final time and return the string with the same format as the initial one
    return f"{minutes:02}:{seconds:02}.{deciseconds}"


def read_data():
    print("--------------------------------------------------------------------------------------------------")
    global data
    data = pd.read_csv('res/partA_input_data.csv')
    display_data(data)
    input("Press Enter to continue...")


def lap_search():
    print("--------------------------------------------------------------------------------------------------")
    while True:
        try:
            limit = int(input(f"Enter the limit of laps to search by (0 - {data['LAPS'].max()}): "))
            if limit < 0:
                raise Exception
            break
        except (ValueError, Exception):
            print("\nInvalid input. Please try again!")

    display_data(data[data['LAPS'] >= limit].sort_values('GRAND PRIX'))
    input("Press Enter to continue...")


def avg_lap_time():
    global data
    print("--------------------------------------------------------------------------------------------------")
    # Creating new column and populating it with new time in the same form as the initial one
    data['AVERAGE LAP TIME'] = data['TIME'].apply(time_to_double) / data['LAPS']
    data['AVERAGE LAP TIME'] = data['AVERAGE LAP TIME'].apply(double_to_time)

    # Writing the new data frame into new file and displaying the results by reading from this new file
    data.to_csv('res/partA_output_data.txt', sep=',', index=False)
    display_data(pd.read_csv('res/partA_output_data.txt'))
    input("Press Enter to continue...")


def field_sort():
    print("--------------------------------------------------------------------------------------------------")
    pi.print_columns_choice(data.columns)
    field = input_error_checker("Enter the field to sort by (number): ", '1', str(len(data.columns)))

    pi.print_order_choice()
    order = input_error_checker("Enter the order (number): ", '1', '2')

    pattern = r'^\d{2}-[A-Z][a-z]{2}-\d{2}$'
    if re.match(pattern, str(data.iloc[:, field-1].iloc[0])):
        display_data(data.sort_values(data.iloc[:, field - 1].name, ascending=order == 1,
                                      key=lambda x: pd.to_datetime(x, format='%d-%b-%y')))
    else:
        display_data(data.sort_values(data.iloc[:, field-1].name, ascending=order == 1))
    input("Press Enter to continue...")


def column_graph():
    print("--------------------------------------------------------------------------------------------------")
    total_average_lap_time = data
    total_average_lap_time['AVERAGE LAP TIME'] = total_average_lap_time['AVERAGE LAP TIME'].apply(time_to_double) / 100
    total_average_lap_time = total_average_lap_time.groupby('WINNER')['AVERAGE LAP TIME'].mean()

    plt.bar(total_average_lap_time.index, total_average_lap_time.values)
    plt.xlabel('Driver Name')
    plt.ylabel('Average Lap Time (minutes)')
    plt.title('Total Average Lap Time per Driver')
    plt.show()

    input("Press Enter to continue...")


def main():
    print("=================================================================")
    print("F1 GRAND PRIX RACING DATA & STATISTICS FOR THE 2023 RACING SEASON")
    print("=================================================================")

    while True:
        print("--------------------------------------------------------------------------------------------------")
        pi.display_menu_instructions()
        option = input("Enter your choice: ").replace(" ", "")

        if option != '1' and option != '6' and data.empty:
            print("\nNot enough data to proceed! Please run \"1\" option to load the data first.")
            input("Press Enter to continue...")
        else:
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
                    if 'AVERAGE LAP TIME' not in data.columns:
                        print("\nNot enough data to proceed. Please execute option 3 first.")
                        input("Press Enter to continue...")
                        continue
                    field_sort()
                case '5':
                    column_graph()
                case '6':
                    print("-------------------------------------------------------------------------------------------"
                          "-------")
                    print("You selected option 6.\nIt's time to say goodbye then...\nBye!")
                    quit(0)
                case default:
                    print("Invalid choice. Please try again!")
                    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
