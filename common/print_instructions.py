# Function to display the menu options for PartA
def display_menu_instructions():
    print("""
1. Read and display the F1 Grand Prix data for the 2023 racing season
2. Filter and sort race data based on a minimum threshold of laps
3. Calculate average lap time per race, save, retrieve, display
4. Sort and display the data based on user parameters
5. Calculate and graph total lap time per driver
6. Exit the program
Hint: Enter \"0\" for more detailed description of each menu option.
        """)


# Function to display the legend for PartA
def print_legend():
    print("--------------------------------------------------------------------------------------------------")
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


# Function to display the columns list to choose from for the sorting function
def print_columns_choice(columns):
    print("\nThe columns options:\n")

    for index, name in enumerate(columns, start=1):
        print(f"{index}. {name}")
    print()


# Function to print the sort order choices
def print_order_choice():
    print("""
The order options:
1. Ascending
2. Descending
            """)


# Function to display the menu options for PartB
# TODO(adjust the options)
def mlp_menu_instructions():
    print("""
1. Read the labelled text data file, display the first 5 lines
2. Choose the size of the hidden layers of the MLP topology (e.g. 6-?-?-2)
3. Choose the size of the training step (0.001 - 0.5, [ENTER] for adaptable)
4. Train on 80% of labeled data, display progress graph
5. Classify the unlabeled data, output training report and confusion matrix
6. Exit the program
Hint: Enter \"0\" for more detailed description of each menu option.
        """)


# Function to display the legend for PartB
# TODO(adjust as well)
def mlp_legend():
    print("--------------------------------------------------------------------------------------------------")
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
