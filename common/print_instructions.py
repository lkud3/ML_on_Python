def display_menu_instructions():
    print("""
1. Read and display the F1 Grand Prix data for the 2023 racing season
2. Filter and sort race data based on a minimum threshold of laps
3. Calculate average lap time per race, save, retrieve, display
4. Sort and display the data based on user parameters
5. Calculate and graph total lap time per driver
6. Exit the program
Hint: Enter (0) for more detailed description of each menu option.
        """)


def print_legend():
    print("""
1. Reads the 6 columns of data from file partA_input_data.txt and neatly displays it on screen.
2. Asking user for a limit of laps to search by, then displays only the race results
which involve that number of home laps or greater, sorted alphabetically by Grand Prix name. 
3. Calculates the average lap time per race then saves this new information as a
7th column in file partA_output_data.txt and displays the new data.
4. Asks the user for a field to sort by and displays on screen all data contained in the file 
sorted according to the user's instructions (ascending or descending).
5. Calculates the total average lap time per driver across all Grand Prix races and
presents it as a GUI column graph in a pop-up window.
6. Exit the program
        """)


def print_columns_choice():
    print("""
The columns options:
1. GRAND PRIX
2. DATE
3. WINNER
4. CAR
5. LAPS
6. TIME
7. AVERAGE LAP TIME
            """)


def print_order_choice():
    print("""
The order options:
1. Ascending
2. Descending
            """)
