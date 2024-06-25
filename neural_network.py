import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import statistic_tests as hp
import data_parser as dp

def nn(df):
    """
    Trains and evaluates a neural network on the provided DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame with 'Tag' column as labels.

    Prints:
    Test loss and accuracy, classification report.
    """
    X = df.drop(columns=['Tag'])
    y = df['Tag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.1, verbose=0)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Generate classification report
    print(classification_report(y_test, y_pred))

def process_folders(base_directory):
    """
    Processes each folder in the base directory, performing neural network training and evaluation.

    Parameters:
    base_directory (str): The base directory containing data folders.
    """
    for folder_name in os.listdir(base_directory):
        yes_folder = os.path.join(base_directory, folder_name, "yes")
        no_folder = os.path.join(base_directory, folder_name, "no")

        print(f'{folder_name}:')
        if os.path.isdir(yes_folder) and os.path.isdir(no_folder):
            df_full = dp.complete_data_to_df(yes_folder, no_folder)
            nn(df_full)

            df_signi_pat = dp.complete_data_to_signi_pat(yes_folder, no_folder, hp.perform_ks_test)
            if df_signi_pat.shape[1] > 1:
                pass

def main():
    """
    Main function to process folders in the specified base directory.
    """
    base_directory = "/plasma_markers"
    process_folders(base_directory)

if __name__ == "__main__":
    main()
