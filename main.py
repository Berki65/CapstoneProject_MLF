import pandas as pd
# Load the dataset
dataset_train = pd.read_csv("PM_train.csv")
dataset_test = pd.read_csv("PM_test.csv")

# Shift the cycle column forward by 1 to identify the last cycle before it resets
dataset_train['Y'] = (dataset_train['cycle'] == dataset_train.groupby('id')['cycle'].transform('max')).astype(int)
dataset_test['Y'] = (dataset_test['cycle'] == dataset_test.groupby('id')['cycle'].transform('max')).astype(int)

# Define selected columns, including 'Y'
selected_columns = ['id', 'cycle', 'setting1', 'setting2', 's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21', 'Y']

# Select the subset of data
selected_dataset_train = dataset_train[selected_columns]
selected_dataset_test = dataset_test[selected_columns]

## Save the selected dataset to an Excel file
# selected_dataset.to_excel("selected_dataset_with_Y.xlsx", index=False)

# Sort by 'id' and 'cycle' to ensure the data is in the correct order
dataset_train = dataset_train.sort_values(['id', 'cycle'])
dataset_test = dataset_test.sort_values(['id', 'cycle'])

# Group by 'id' and select the last 10 cycles for each 'id'
last_10_cycles_train = dataset_train.groupby('id').tail(10)
last_10_cycles_test = dataset_test.groupby('id').tail(10)

# Further filter the selected columns for this subset
selected_last_10_cycles_train = last_10_cycles_train[selected_columns]
selected_last_10_cycles_test = last_10_cycles_test[selected_columns]

# Display the result
selected_last_10_cycles_train.head()
selected_last_10_cycles_test.head()

# Define the features (X) and target (Y)
X_train = selected_dataset_train.drop(columns=['Y', 'id', 'cycle'])  # Drop 'Y', 'id', and 'cycle' if they are not part of the model
Y_train = selected_dataset_train['Y']
X_test = selected_dataset_test.drop(columns=['Y', 'id', 'cycle'])  # Drop 'Y', 'id', and 'cycle' if they are not part of the model
Y_test = selected_dataset_test['Y']

# Alternatively, if using only the last 10 cycles:
X_last_10_train = selected_last_10_cycles_train.drop(columns=['Y', 'id', 'cycle'])
Y_last_10_train = selected_last_10_cycles_train['Y']
X_last_10_test = selected_last_10_cycles_test.drop(columns=['Y', 'id', 'cycle'])
Y_last_10_test = selected_last_10_cycles_test['Y']
