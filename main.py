# SALTYKOV DANIIL 07/05/2024
# F1 RACING RESULTS
#
# Copyright: GNU Public License http://www.gnu.org/licenses/
# 20220372@student.act.edu

def menu_option_check(value, min_option, max_option):
    if ascii(value) < ascii(min_option) or ascii(value) > ascii(max_option):
        print(f"Invalid input! Please make sure your option choice is between {min_option} and {max_option}.")
        return False
    return True  # Thing to discuss


def display_menu_instructions():
    print("""
1. Read and display the F1 Grand Prix data for the 2023 racing season
2. Filter and sort race data based on a minimum threshold of laps
3. Calculate average lap time per race, save, retrieve, display
4. Sort and display the data based on user parameters
5. Calculate and graph total lap time per driver
6. Exit the program
        """)


def switch_choice(option):
    match option:
        case 1:
            read_data()
        case 2:
            lap_search()
        case 3:
            avg_lap_time()
        case 4:
            field_sort()
        case 5:
            column_graph()
        case 6:
            print("""
                You selected option 6.
                It's time to say goodbye then...
                Bye!
            """)
            quit(0)
        case default:
            print("Error occurred")


def read_data():
    print("-----------------------------------------------------------------")


def lap_search():
    print("-----------------------------------------------------------------")


def avg_lap_time():
    print("-----------------------------------------------------------------")


def field_sort():
    print("-----------------------------------------------------------------")


def column_graph():
    print("-----------------------------------------------------------------")


def main():
    print("=================================================================")
    print("F1 GRAND PRIX RACING DATA & STATISTICS FOR THE 2023 RACING SEASON")
    print("=================================================================")

    option = 0
    display_menu_instructions()


if __name__ == "__main__":
    main()
