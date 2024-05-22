# SALTYKOV DANIIL 07/05/2024
# F1 RACING RESULTS
#
#
#
# Copyright: GNU Public License http://www.gnu.org/licenses/
# 20220372@student.act.edu

from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import common.print_instructions as pi

# Initializing the global data frame for data storage
data = pd.DataFrame()
column_names = ['GRAND PRIX', 'DATE', 'WINNER', 'CAR', 'LAPS', 'TIME', 'AVERAGE LAP TIME']


def menu_option_check(value, min_option, max_option):
    if ascii(value) < ascii(min_option) or ascii(value) > ascii(max_option):
        print(f"Invalid input! Please make sure your option choice is between {min_option} and {max_option}.")
        return False
    return True  # Thing to discuss


def display_data(data_input) -> None:
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
    " " + input("Press Enter to continue...")


def lap_search():
    print("--------------------------------------------------------------------------------------------------")

    while True:
        try:
            limit = int(input("Enter the limit of laps to search by (0-78): "))
        except ValueError:
            print("\nInvalid input. Please try again!")
        else:
            if 0 <= limit <= 78:
                break
            else:
                print("\nThe limit you entered is out of range. Please revaluate your choice!")

    display_data(data[data['LAPS'] >= limit].sort_values('GRAND PRIX'))
    " " + input("Press Enter to continue...")


def avg_lap_time():
    global data
    print("--------------------------------------------------------------------------------------------------")
    # Creating new column and populating it with new time in the same form as the initial one
    data['AVERAGE LAP TIME'] = data['TIME'].apply(time_to_double) / data['LAPS']
    data['AVERAGE LAP TIME'] = data['AVERAGE LAP TIME'].apply(double_to_time)

    # Writing the new data frame into new file and displaying the results by reading from this new file
    data.to_csv('res/partA_output_data.txt', sep=',', index=False)
    display_data(pd.read_csv('res/partA_output_data.txt'))
    " " + input("Press Enter to continue...")


def field_sort():
    print("--------------------------------------------------------------------------------------------------")

    # TODO(may be rework the choice, to make it just numbers)
    pi.print_columns_choice()
    while True:
        field = input("Enter the field to sort by (name of the column): ")
        if field.upper() in column_names:
            break
        print("\nThere is no such a column in data. Check the spelling!")

    pi.print_order_choice()
    while True:
        order = input("Enter the order (asc or desc): ")
        if order.lower() in ['asc', 'desc']:
            break
        print("\nYou entered invalid order. Please revaluate your choice!")

    # TODO(Sorting of date)
    display_data(data.sort_values(field.upper(), ascending=order.lower() == 'asc'))
    " " + input("Press Enter to continue...")


def column_graph():
    print("--------------------------------------------------------------------------------------------------")
    " " + input("Press Enter to continue...")


def main():
    print("=================================================================")
    print("F1 GRAND PRIX RACING DATA & STATISTICS FOR THE 2023 RACING SEASON")
    print("=================================================================")

    while True:
        print("--------------------------------------------------------------------------------------------------")
        pi.display_menu_instructions()
        option = input("Enter your choice: ")

        # TODO("Restrictions for menu option until 1 is implemented")
        # if not choice == "1" and not choice == "6" and table is None:
        #     print("Please use choice \"1\" to load the table first.")
        #     return

        match option:
            case '0':
                pi.print_legend()
                " " + input("Press Enter to continue...")
            case '1':
                read_data()
            case '2':
                lap_search()
            case '3':
                avg_lap_time()
            case '4':
                if 'AVERAGE LAP TIME' not in data.columns:
                    print("\nNot enough data. Please execute option 3 first.")
                    " " + input("Press Enter to continue...")
                    continue
                field_sort()
            case '5':
                column_graph()
            case '6':
                print("----------------------------------------------------------------------------------------------"
                      "----")
                print("You selected option 6.\nIt's time to say goodbye then...\nBye!")
                quit(0)
            case default:
                print("Invalid choice. Please try again!")
                " " + input("Press Enter to continue...")


if __name__ == "__main__":
    main()
