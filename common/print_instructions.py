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
2. Choose the size of the hidden layers of the MLP topology (e.g. 4-?-?-1)
3. Choose the size of the training step (0.001 - 0.5, [ENTER] for adaptable)
4. Train on 80% of labeled data, display progress graph
5. Classify the unlabeled data, output training report and confusion matrix
6. Make 18 tests to analyze the best approach
7. Exit the program
Hint: Enter \"0\" for more detailed description of each menu option.
        """)


# Function to display the legend for PartB
def mlp_legend():
    print("--------------------------------------------------------------------------------------------------")
    print("""
1. Reads the data from file and, parsing it and preparing for future actions. Prints the first 5 lines of the dataframe.
2. Asking user for a hidden layers topology of the MLP. In case of empty input, make the default reasonable topology.
3. Asking user for a training step the MLP to adapt with. In case of empty input, make the training step adaptable. 
4. Splits the data on 80/20 to train on 80% of the data and use the rest of 20% to test the predictions. 
Shows the progress graph.
5. Uses the 20% of data to make prediction for classification. Shoes the results, including training report and 
confusion matrix.
6. Make a tests of the MLP with 3 different topologies, 3 different learning steps and 2 different training approaches
to evaluate the best combination out of 18 of them. Saves all the error graphs created into a plots folder.
7. Exit the program
        """)
