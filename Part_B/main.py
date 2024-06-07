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
import numpy as np
import matplotlib.pyplot as plt
import common.print_instructions as pi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# TODO(save all the new data into files)


def read_data(file, names):
    print("--------------------------------------------------------------------------------------------------")
    print(f"Reading the data from file {file}...")
    data = pd.read_csv(file, names=names)
    print("\nPrinting the first few lines of the data set:\n", tabulate(data.head(), headers='keys'))

    data_variables = data.iloc[:, 0:4]
    data_labels = data.select_dtypes(include=['int64'])
    print("\nPrinting the labels column categories: ", data_labels.Class.unique(), "\n")

    input("Press Enter to continue...")
    return data_variables, data_labels


# TODO(add default topology)
def topology_choice():
    print("--------------------------------------------------------------------------------------------------")
    while True:
        try:
            # Receives user input and catches non-integer input
            input_string = input("Enter the size of the hidden layers of the MLP topology (e.g. 10-10-10).\nHINT: "
                                 "input and output layers are fixed, choose the size of hidden"
                                 " layers only: ").replace(" ", "")

            topology = tuple(int(x) for x in input_string.split("-") if x != "")

            if len(topology) < 2:
                raise ValueError("The MLP should have at least two hidden layers. Please try again!")

            if any(x == 0 for x in topology):
                raise ValueError("The hidden layer can not consist of zero nodes. Please try again!")

            print(f"\nThe topology for hidden layers {input_string} has been chosen.")
            input("Press Enter to continue...")
            return topology

        except ValueError as e:
            print(f"\nError: {e}\n")


def training_step_choice():
    print("--------------------------------------------------------------------------------------------------")
    while True:
        # Receives user input and catches non-integer input
        step = input(f"Enter the size of the training step (0.001 - 0.5, [ENTER] for adaptable): ").replace(" ", "")

        if step == "":
            print(f"\nThe learning step will be adaptable.")
            input("Press Enter to continue...")
            return 'adaptable'

        try:
            step = float(step)

            if 0.001 <= step <= 0.5:
                print(f"\nThe learning step of {step} has been chosen.")
                input("Press Enter to continue...")
                return step
            else:
                print("\nThe learning step should be between 0.001 and 0.5. Please try again!")

        except ValueError:
            print("\nInvalid input! Please enter a valid double value. Please try again!")


def plot_training_error(mlp, entry_train, labels_train, epochs):
    print("The error graph is building...")
    errors = []
    for i in range(epochs):
        mlp.partial_fit(entry_train, labels_train.values.ravel())
        errors.append(mlp.loss_)
    plt.plot(range(epochs), errors)
    plt.xlabel('Epochs')
    plt.ylabel('Training Error')
    plt.title('Training Error Curve')
    print("\nThe graph is on the screen. Please close the window in order to proceed...")
    plt.show()


def train_mlp(entry_train, labels_train, topology, step):
    print("--------------------------------------------------------------------------------------------------")
    if isinstance(step, str):
        mlp = MLPClassifier(hidden_layer_sizes=topology, max_iter=1000, learning_rate='adaptive')
    else:
        mlp = MLPClassifier(hidden_layer_sizes=topology, max_iter=1000, learning_rate_init=step)

    mlp.fit(entry_train, labels_train.values.ravel())

    plot_training_error(mlp, entry_train, labels_train, 1000)

    print("\nThe mlp is trained.")
    input("Press Enter to continue...")

    return mlp


def experiments_statistics():
    hidden_layers_options = [(6, 6), (6, 12), (12, 6), 6]
    learning_rate_options = [0.001, 0.5, 'adaptable']
    split_options = [(0.5, False), (0.8, True)]

    for hidden_layers in hidden_layers_options:
        for learning_rate in learning_rate_options:
            for split in split_options:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split[0],
                                                                    random_state=42 if split[1] else None)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                mlp = train_mlp(X_train, y_train, hidden_layers, learning_rate)
                accuracy, cm = evaluate_mlp(mlp, X_test, y_test)
                print(f"Hidden Layers: {hidden_layers}, Learning Rate: {learning_rate}, Split: {split[0]}")
                print(f"Accuracy: {accuracy:.3f}, Confusion Matrix:\n{cm}")
                plot_training_error(mlp, X_train, y_train, 100)


def mpl_prediction(mlp, values_test, labels_test) -> None:
    print("--------------------------------------------------------------------------------------------------")
    predictions = mlp.predict(values_test)

    print('\nNow printing the confusion matrix (without normalization)...\n')
    print(confusion_matrix(labels_test, predictions))
    print('\nNow printing the classification report...\n')
    print(classification_report(labels_test, predictions))
    print('\nNow printing the accuracy score...')
    print(accuracy_score(labels_test, predictions))
    input("Press Enter to continue...")


# The main function
def main():
    file = 'res/data_banknote_authentication.txt'
    names = ['variance', 'skewness', 'curtosis', 'entropy', 'Class']
    labels = pd.DataFrame()
    entry_data = pd.DataFrame()
    entry_test = pd.DataFrame()
    labels_test = pd.DataFrame()
    topology = tuple()
    step = 0.0

    print("=======================================================================================")
    print("USER MENU: MLP CLASSIFICATION OF THE BANK NOTE IDENTIFICATION DATA SET (UCI REPOSITORY)")
    print("=======================================================================================")

    while True:
        print("--------------------------------------------------------------------------------------------------")
        pi.mlp_menu_instructions()
        option = input("Enter your choice: ").replace(" ", "")

        # Checks if the user use option 1 first to read the data
        if option != '1' and option != '7' and labels.empty:
            print("\nNot enough data to proceed! Please run \"1\" option to load the data first.")
            input("Press Enter to continue...")

        # TODO(make limitations for option 4 (calling before 2 and 3), 5 should go after 4 )
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
                    entry_train, entry_test, labels_train, labels_test = train_test_split(entry_data, labels,
                                                                                          test_size=0.20)
                    scaler = StandardScaler()
                    scaler.fit(entry_train)

                    entry_train = scaler.transform(entry_train)
                    entry_test = scaler.transform(entry_test)

                    mlp = train_mlp(entry_train, labels_train, topology, step)
                case '5':
                    mpl_prediction(mlp, entry_test, labels_test)
                case '6':
                    print()
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
