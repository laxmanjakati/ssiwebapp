# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np

# Set title and banner
st.set_page_config(page_title="Project Work", page_icon=":microphone:", layout="wide", initial_sidebar_state="collapsed")

# Define the available models
MODELS = {
    "Random Forest": RandomForestClassifier(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbor": KNeighborsClassifier(n_neighbors=5)
}

# Define a function to load a saved model
def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

# Define a function to predict labels and calculate accuracy, precision, recall, and F1 score
def predict_and_evaluate(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    label_accuracies = {}
    for label in set(y):
        label_indices = [i for i, x in enumerate(y) if x == label]
        label_y = y[label_indices]
        label_y_pred = y_pred[label_indices]
        label_accuracy = accuracy_score(label_y, label_y_pred)
        label_accuracies[label] = label_accuracy
    return y_pred, accuracy, precision, recall, f1, label_accuracies

# Load the EMG data set into a pandas dataframe
emg_data = pd.read_csv('/content/train_data.csv')

# Split the data set into features (X) and labels (y)
X = emg_data.iloc[:, 1:]
y = emg_data.iloc[:, 0]

# Train and save the available models
for name, model in MODELS.items():
    # Train the model on the EMG data
    model.fit(X, y)
    # Save the trained model as a pickle file
    filename = f"{name.lower().replace(' ', '_')}_model.pkl"
    pickle.dump(model, open(filename, 'wb'))

# Get the user's selection of model
model_name = st.sidebar.selectbox("Select a model", list(MODELS.keys()))

# Load the selected model
model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
loaded_model = load_model(model_filename)

# Define the sidebar image
st.sidebar.image("/content/sidebar_image.png", use_column_width=True)

# Display the headers
st.markdown("<h1 style='text-align: center; color: #14425A;'>Silent Speech Interface of EMG Signals</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #14425A;'>Govt SKSJTI, ECE DEPT</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #14425A;'>Project Team 09: Ananad, Kiran Kumar S, Lakshmana and Mahesh S</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #14425A;'>Guided by Dr. N. Sathisha</h4>", unsafe_allow_html=True)

# Define the instructions section
st.sidebar.title("Instructions")
st.sidebar.write("1. Select a model from the dropdown list.")
st.sidebar.write("2. Upload a CSV file containing test data.")
st.sidebar.write("3. Wait for the prediction results and test metrics to appear.")
st.sidebar.write("4. Check the highlighted rows in the prediction results table for incorrect predictions.")
st.sidebar.write("5. View the train metrics and prediction results on the training dataset.")

# Get the test data from user
st.sidebar.subheader("Upload Test Data")
test_data = st.sidebar.file_uploader("Choose a CSV file containing test data", type="csv")

# Check if test data is uploaded
if test_data is not None:
    # Load the test data set into a pandas dataframe
    new_data = pd.read_csv(test_data)

    # Extract features and labels from the new dataset
    X_new = new_data.iloc[:, 1:]
    y_new = new_data.iloc[:, 0]

    # Predict labels for the new dataset and calculate accuracy, precision, recall, and F1 score
    y_pred_new, accuracy_new, precision_new, recall_new, f1_new, label_accuracies_new = predict_and_evaluate(loaded_model, X_new, y_new)

    # Create a dataframe for the prediction results
    results_data = {
        'Actual Labels': y_new,
        'Predicted Labels': y_pred_new,
        'Correct Prediction': np.where(y_new == y_pred_new, 'Yes', 'No')
    }
    results_df = pd.DataFrame(results_data)

    # Highlight the rows where the predicted label is different from the actual label
    highlight_color = '#FFCDD2'  # Light red
    incorrect_pred_mask = results_df['Correct Prediction'] == 'No'
    results_df = results_df.style.applymap(lambda x: f'background-color: {highlight_color}', subset=pd.IndexSlice[incorrect_pred_mask, ['Predicted Labels']])

    # Display the prediction results
    st.subheader("Prediction Results")
    st.write(results_df)
    st.subheader("Test Metrics")
    st.write("- Accuracy: {:.2f}%".format(accuracy_new * 100))
    st.write("- Precision: {:.2f}".format(precision_new))
    st.write("- Recall: {:.2f}".format(recall_new))
    st.write("- F1 score: {:.2f}".format(f1_new))
else:
    st.sidebar.warning("Please upload test data file to see prediction results and test metrics.")

# Display train accuracy
_, accuracy_train, precision_train, recall_train, f1_train, label_accuracies_train = predict_and_evaluate(loaded_model, X, y)
st.subheader("Train Metrics")
st.write("- Accuracy on training dataset: {:.2f}%".format(accuracy_train * 100))
st.write("- Precision on training dataset: {:.2f}".format(precision_train))
st.write("- Recall on training dataset: {:.2f}".format(recall_train))
st.write("- F1 score on training dataset: {:.2f}".format(f1_train))
train_results_df = pd.DataFrame({'Actual Labels': y, 'Predicted Labels': loaded_model.predict(X)})
st.write(train_results_df)

# Display main screen image
main_image = "/content/mainscreen.jpeg"
st.image(main_image, use_column_width=True)