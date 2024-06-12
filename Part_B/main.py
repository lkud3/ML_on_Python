# SALTYKOV DANIIL 27/05/2024
# MLP Classification
#
# This project implements a small deterministic multi-layer perceptron (MLP) artificial
# neural network (ANN), consisting of binary state neurons, analogue state weights and a variable
# topology. It implements a supervised training
# algorithm, back-propagation, reading training data from a file which contains both the input
# binary vectors as well as the desired classification labels.
# It was validated using the data sets of banknote authentication.
#
# The interface makes it possible for the user to load the data, choose the topology of hidden layers and
# training step of the MLP, to train it on live and perform testing evaluation the part of initial dataset.
#
# Additionally, the 6 option provides the statistics of different MLP configurations and comparative analysis of the
# error and weights changes.
#
# Copyright: GNU Public License http://www.gnu.org/licenses/
# 20220372@student.act.edu

# Import of required libraries
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import common.print_instructions as pi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Function to read the data from the file
def read_data(file, names):
    print("--------------------------------------------------------------------------------------------------")
    print(f"Reading the data from file {file}...")

    # Reads the data and displays it in appropriate form
    data = pd.read_csv(file, names=names)
    print("\nPrinting the first few lines of the data set:\n", tabulate(data.head(), headers='keys'))

    # Separates the variables and labels for future use
    data_variables = data.iloc[:, 0:4]
    data_labels = data.select_dtypes(include=['int64'])
    print("\nPrinting the labels column categories: ", data_labels.Class.unique(), "\n")

    input("Press Enter to continue...")
    return data_variables, data_labels


# This function let the user choose the topology of hidden layers
def topology_choice():
    print("--------------------------------------------------------------------------------------------------")
    while True:
        input_string = input("Enter the size of the hidden layers of the MLP topology (e.g. 10-10-10), [ENTER] "
                             "for default) \nHINT: "
                             "input and output layers are fixed, choose the size of hidden"
                             " layers only: ").replace(" ", "")

        # Makes a default topology of 3-3 in case the input is empty
        if input_string == "":
            print(f"\nThe topology for hidden layers will be 3-3.")
            input("Press Enter to continue...")
            topology = (3, 3)
            return topology

        # Catches the incorrect input
        try:
            topology = tuple(int(x) for x in input_string.split("-") if x != "")

            # Validate the input
            if len(topology) < 2:
                raise ValueError("The MLP should have at least two hidden layers. Please try again!")

            if any(x == 0 for x in topology):
                raise ValueError("The hidden layer can not consist of zero nodes. Please try again!")

            print(f"\nThe topology for hidden layers {input_string} has been chosen.")
            input("Press Enter to continue...")
            return topology

        except ValueError as e:
            print(f"\nError: {e}\n")


# This function let the user choose the learning step
def training_step_choice():
    print("--------------------------------------------------------------------------------------------------")
    while True:
        # Receives user input and catches non-integer input
        step = input(f"Enter the size of the training step (0.001 - 0.5, [ENTER] for adaptable): ").replace(" ", "")

        # Makes a default topology of adaptable method in case the input is empty
        if step == "":
            print(f"\nThe learning step will be adaptable.")
            input("Press Enter to continue...")
            return 'adaptable'

        # Catches the incorrect input
        try:
            step = float(step)

            # Validates the input
            if 0.001 <= step <= 0.5:
                print(f"\nThe learning step of {step} has been chosen.")
                input("Press Enter to continue...")
                return step
            else:
                print("\nThe learning step should be between 0.001 and 0.5. Please try again!")

        except ValueError:
            print("\nInvalid input! Please enter a valid double value. Please try again!")


# This function is building the training error graph, both for main menu and statistical iterations
def plot_training_error(mlp, entry_train, labels_train, title, file_name="", to_show=True):
    # The first mlp fit should be with initial classes so we create them
    classes = np.unique(labels_train.values.ravel())
    errors = []
    # Makes 300 repetitions to record 300 of final error
    for i in range(300):
        if i == 0:
            mlp.partial_fit(entry_train, labels_train.values.ravel(), classes=classes)
        else:
            mlp.partial_fit(entry_train, labels_train.values.ravel())
        errors.append(mlp.loss_)

    # Makes the graph
    plt.plot(range(300), errors)
    plt.xlabel('Epochs')
    plt.ylabel('Training Error')
    plt.title(title)
    # If the function is called from the menu, the graph is displayed
    if to_show:
        print("\nThe graph is on the screen. Please close the window in order to proceed...")
        plt.show()
        plt.close()
    # in other case, it is just saved in the folder
    else:
        plt.savefig(file_name)
        plt.close()
        return errors


# This function makes the weight changing graph
def plot_weight_changes(mlp, entry_train, labels_train, title, file_name):
    weights = []
    classes = np.unique(labels_train.values.ravel())
    for i in range(10):
        if i == 0:
            mlp.partial_fit(entry_train, labels_train.values.ravel(), classes=classes)
        else:
            mlp.partial_fit(entry_train, labels_train.values.ravel())
        # appending the weights
        weights.append(mlp.coefs_[0].copy())
    # building a plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    for w in weights:
        plt.plot(w.flatten())
    plt.xlabel('Epochs')
    plt.ylabel('Weights')
    plt.title(title)
    plt.savefig(file_name)


# This function classifies the mlp
def mlp_classifier(topology, step):
    # makes it with adaptable or set in training step
    if isinstance(step, str):
        mlp = MLPClassifier(hidden_layer_sizes=topology, max_iter=1000, learning_rate='adaptive')
    else:
        mlp = MLPClassifier(hidden_layer_sizes=topology, max_iter=1000, learning_rate_init=step)

    return mlp


# This function performs the comprehensive comparative statistics building mlp with multiple configurations
def experiments_statistics(entry_data, labels):
    print("--------------------------------------------------------------------------------------------------")
    # All the configurations to go through
    hidden_layers_options = [(2, 2), (3, 6), 6, 3]
    learning_rate_options = [0.001, 0.5, 'adaptable']
    split_options = [(0.5, True), (0.2, False)]
    layer_counter = 0
    all_errors = []

    # the loop to try all combinations
    for hidden_layers in hidden_layers_options:
        layer_counter += 1
        rate_counter = 0
        for learning_rate in learning_rate_options:
            rate_counter += 1
            split_counter = 0
            for split in split_options:
                split_counter += 1
                # Splitting the data
                entry_train, entry_test, labels_train, labels_test = train_test_split(entry_data,
                                                                                      labels, test_size=split[0],
                                                                                      shuffle=False if split[1]
                                                                                      else True)

                # Making zero mean
                scaler = StandardScaler()
                scaler.fit(entry_train)

                entry_train = scaler.transform(entry_train)

                # Classifying
                mlp = mlp_classifier(hidden_layers, learning_rate)

                # Makes two weight changing plots
                if (hidden_layers == 6 and learning_rate == 0.5 and split[0] == 0.2) \
                        or (hidden_layers == (3, 6) and learning_rate == 0.001 and split[0] == 0.2):
                    print(
                        f"Building Weights Changing Graph - Hidden Layers: {hidden_layers}, Learning Rate: {learning_rate},"
                        f" Split: {split[0]}")
                    title = (f"Weights Changing Graph - Hidden Layers: {hidden_layers}, Learning Rate: {learning_rate}, "
                             f"Split: {split[0]}")
                    file_name = f"plots/weight_changes/weight_changing_{layer_counter}_{rate_counter}_{split_counter}.png"
                    plot_weight_changes(mlp, entry_train, labels_train, title, file_name)

                # Building the Training Error Curve for every configuration
                print(f"Building Training Error Curve - Hidden Layers: {hidden_layers}, Learning Rate: {learning_rate},"
                      f" Split: {split[0]}")
                title = (f"Training Error Curve - Hidden Layers: {hidden_layers}, Learning Rate: {learning_rate}, "
                         f"Split: {split[0]}")
                file_name = f"plots/training_error/training_error_{layer_counter}_{rate_counter}_{split_counter}.png"

                all_errors.append(plot_training_error(mlp, entry_train, labels_train, title, file_name, False))

    print("\nAll the error graphs are build successfully")
    print("\nBuilding the Collective Error Curve...")

    # Builds the collective error graph
    plt.figure(figsize=(8, 6))
    for i, errors in enumerate(all_errors):
        plt.plot(range(300), errors,
                 label=f"Hidden Layers: {hidden_layers_options[i // 6]}, Learning Rate: "
                       f"{learning_rate_options[i % 6 // 2]}, Split: {split_options[i % 2][0]}")
    plt.xlabel('Epochs')
    plt.ylabel('Training Error')
    plt.title('Collective Error Curve')
    plt.legend(fontsize=8, ncol=2)
    plt.savefig("plots/collective_error_graph.png")

    print("\nAll the graphs are build successfully")
    input("Press Enter to continue...")


# This function makes a test of prediction using mlp
def mpl_prediction(mlp, values_test, labels_test, entry_test_for_output) -> None:
    print("--------------------------------------------------------------------------------------------------")
    # makes prediction
    predictions = mlp.predict(values_test)

    # Calculates and prints the evaluated statistics
    print('\nNow printing the confusion matrix (without normalization)...\n')
    print(confusion_matrix(labels_test, predictions))
    print('\nNow printing the classification report...\n')
    print(classification_report(labels_test, predictions))
    print('\nNow printing the accuracy score...\n')
    print(accuracy_score(labels_test, predictions))

    names = ['variance', 'skewness', 'curtosis', 'entropy', 'Predictions']
    predictions_df = pd.DataFrame(data=np.hstack((entry_test_for_output, predictions.reshape(-1, 1))),
                                  columns=names)

    predictions_df.to_csv('output/output_data.txt', index=False)

    input("\nPress Enter to continue...")


# The main function
def main():
    # Variables initialization
    file = 'res/data_banknote_authentication.txt'
    names = ['variance', 'skewness', 'curtosis', 'entropy', 'Class']
    labels = pd.DataFrame()
    entry_data = pd.DataFrame()
    entry_test = pd.DataFrame()
    entry_test_for_output = pd.DataFrame()
    labels_test = pd.DataFrame()
    topology = tuple()
    step = 0
    mlp = 0

    print("=======================================================================================")
    print("USER MENU: MLP CLASSIFICATION OF THE BANK NOTE IDENTIFICATION DATA SET (UCI REPOSITORY)")
    print("=======================================================================================")

    while True:
        print("--------------------------------------------------------------------------------------------------")
        pi.mlp_menu_instructions()
        option = input("Enter your choice: ").replace(" ", "")

        # Checks if the user use option 1 first to read the data
        if option != '0' and option != '1' and option != '7' and labels.empty:
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
                    entry_data, labels = read_data(file, names)
                case '2':
                    topology = topology_choice()
                case '3':
                    step = training_step_choice()
                case '4':
                    # Prevents the user to execute the option 4 before setting the learning rate and topology
                    if not topology or step == 0:
                        print("\nNot enough data to proceed. Please execute option 2 and 3 first.")
                        input("Press Enter to continue...")
                        continue

                    print("--------------------------------------------------------------------------------------------"
                          "------")
                    # Splitting the data
                    entry_train, entry_test, labels_train, labels_test = train_test_split(entry_data, labels,
                                                                                          test_size=0.20)

                    # Making zero mean and saving the dataframes
                    train_df = pd.concat([entry_train, labels_train], axis=1)
                    train_df.to_csv('output/training_data.txt', index=False)

                    entry_test.to_csv('output/testing_data_unlabeled.txt', index=False)

                    test_df = pd.concat([entry_test, labels_test], axis=1)
                    test_df.to_csv('output/testing_data_labeled.txt', index=False)

                    entry_test_for_output = entry_test

                    scaler = StandardScaler()
                    scaler.fit(entry_train)

                    entry_train = scaler.transform(entry_train)
                    entry_test = scaler.transform(entry_test)

                    # Classifying
                    mlp = mlp_classifier(topology, step)

                    # Making graph and fitting the mlp
                    print("The error graph is building...")
                    plot_training_error(mlp, entry_train, labels_train, 'Training Error Curve')

                    mlp.fit(entry_train, labels_train.values.ravel())

                    print("\nThe mlp is trained.")
                    input("Press Enter to continue...")
                case '5':
                    # Testing the mlp and preventing from executing it before fitting
                    if mlp == 0:
                        print("\nThe mlp is not trained. Execute option 4 first.")
                        input("Press Enter to continue...")
                    else:
                        mpl_prediction(mlp, entry_test, labels_test, entry_test_for_output)
                case '6':
                    experiments_statistics(entry_data, labels)
                case '7':
                    print("-------------------------------------------------------------------------------------------"
                          "-------")
                    print("You selected option 7.\nIt's time to say goodbye then...\nBye!")
                    quit(0)
                # The default case for wrong data input
                case _:
                    print("Invalid choice. Please try again!")
                    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
