# SALTYKOV DANIIL 27/05/2024
# F1 RACING RESULTS
#
# The aim of this project is to enhance the knowledge of programming on Python. Program displays well detailed menu
# with 5 options to choose from, each of it performs various operations described in separate functions.
# This program is capable of reading and writing the data from/into .csv file, store this data in the dataframe and
# perform different operations with this dataframe. Most of the project was written to be as general and universal as
# it can be, although some aspects were hard-coded. All the user input is correctly handled in order to catch
# incorrect data being typed.
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

# Initializing the global data frame for data storage
data = pd.DataFrame()


# Special function used for user input check, so it would be in the range of potential option choices. It also checks
# the type of entered data.
def input_error_checker(message, min_option, max_option):
    while True:
        # Reads the input and deletes unnecessary spaces
        value = input(message).replace(" ", "")
        # Performs the validation
        if min_option <= value <= max_option:
            return int(value)
        else:
            print(f"\nInvalid input! Please make sure your option choice is between {min_option} and {max_option}.")


# The function to correctly display the data on the screen in case the dataframe is not empty.
def display_data(data_input) -> None:
    if data_input.empty:
        print("\nThere is no data matching your request.")
    else:
        # Using tabulate to print data clearly and tabulated
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


# Function for the first option to read the data from the file and store it into the dataframe
def read_data():
    print("--------------------------------------------------------------------------------------------------")
    global data

    # Checks if file exists
    if not os.path.isfile('res/partA_input_data.csv'):
        print("The file partA_input_data.csv does not exist.")
        input("Press Enter to continue...")
        return

    # Reads the data and displays it
    data = pd.read_csv('res/partA_input_data.csv')
    print("\nThe data for F1 2023 racing season is following: \n")
    display_data(data)
    input("Press Enter to continue...")


# Function for second menu options. Allows the user to search for records with amount of laps more or equal to the
# limit typed by the user. Sorts the results by Grand Prix name.
def lap_search():
    print("--------------------------------------------------------------------------------------------------")
    while True:
        try:
            # Receives user input and catches non-integer input
            limit = int(input(f"Enter the limit of laps to search by (0 - {data['LAPS'].max()}): "))
            if limit < 0:
                raise Exception
            break
        except (ValueError, Exception):
            print("\nInvalid input. Please try again!")

    # Displays the sorted and filtered data
    print("\nAccording to your limit of laps to search by, the results are following: \n")
    display_data(data[data['LAPS'] >= limit].sort_values('GRAND PRIX'))
    input("Press Enter to continue...")


# Function for third menu option. It calculates the time column into double format, calculated the average lap time and
# stores it into new column in the same format as the initial time. Afterward it is written to the new file.
def avg_lap_time():
    global data
    print("--------------------------------------------------------------------------------------------------")
    # Creating new column and populating it with new time in the same form as the initial one
    data['AVERAGE LAP TIME'] = data['TIME'].apply(time_to_double) / data['LAPS']
    data['AVERAGE LAP TIME'] = data['AVERAGE LAP TIME'].apply(double_to_time)

    # Writing the new data frame into new file and displaying the results by reading from this new file
    data.to_csv('res/partA_output_data.txt', sep=',', index=False)

    # Checks if file exists
    if not os.path.isfile('res/partA_output_data.txt'):
        print("The file partA_output_data.txt does not exist.")
        input("Press Enter to continue...")
        return

    # Displays the data
    print("\nNew data with AVERAGE LAP TIME metric is following: \n")
    display_data(pd.read_csv('res/partA_output_data.txt'))
    input("Press Enter to continue...")


# Function for fourth menu option. It asks user for the field and type to sort by, then presents the data in desired f
# format.
def field_sort():
    print("--------------------------------------------------------------------------------------------------")
    # Enters the field to sort by
    pi.print_columns_choice(data.columns)
    field = input_error_checker("Enter the field to sort by (number): ", '1', str(len(data.columns)))

    # Enters the type of sorting (ascending or descending)
    pi.print_order_choice()
    order = input_error_checker("Enter the order (number): ", '1', '2')

    # Checks if the field to sort by looks like a date, then performs it a bit differently to make filtering for the
    # date correctly
    pattern = r'^\d{2}-[A-Z][a-z]{2}-\d{2}$'
    print("\nAccording to your sorting criteria, the results are following: \n")
    if re.match(pattern, str(data.iloc[:, field-1].iloc[0])):
        display_data(data.sort_values(data.iloc[:, field - 1].name, ascending=order == 1,
                                      key=lambda x: pd.to_datetime(x, format='%d-%b-%y')))
    else:
        display_data(data.sort_values(data.iloc[:, field-1].name, ascending=order == 1))
    input("Press Enter to continue...")


# Function for fifth option. It builds the graph based on total average lap time for every driver.
def column_graph():
    print("--------------------------------------------------------------------------------------------------")
    # Creates new dataframe to calculate the total average lap time and group the data by the driver
    total_average_lap_time = data
    total_average_lap_time['AVERAGE LAP TIME'] = total_average_lap_time['AVERAGE LAP TIME'].apply(time_to_double) / 100
    total_average_lap_time = total_average_lap_time.groupby('WINNER')['AVERAGE LAP TIME'].mean()

    print("The graph is build in pop-up window. Please close the window to continue...")

    # Builds the plot based on the drivers and their total ALT
    plt.bar(total_average_lap_time.index, total_average_lap_time.values)
    plt.xlabel('Driver Name')
    plt.ylabel('Average Lap Time (minutes)')
    plt.title('Total Average Lap Time per Driver')
    plt.show()

    input("Press Enter to continue...")


# The main function
def main():
    print("=================================================================")
    print("F1 GRAND PRIX RACING DATA & STATISTICS FOR THE 2023 RACING SEASON")
    print("=================================================================")

    while True:
        print("--------------------------------------------------------------------------------------------------")
        pi.display_menu_instructions()
        option = input("Enter your choice: ").replace(" ", "")

        # Checks if the user use option 1 first to read the data
        if option != '1' and option != '6' and data.empty:
            print("\nNot enough data to proceed! Please run \"1\" option to load the data first.")
            input("Press Enter to continue...")
        else:
            # Match case for menu implementation
            match option:
                # Additional case for more detailed information about menu options
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
                    # Checks if the option 3 was used before option 4
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
                # The default case for wrong data input
                case _:
                    print("Invalid choice. Please try again!")
                    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
