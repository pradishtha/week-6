import pandas as pd
import streamlit as st
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Title of the Streamlit app
st.title("COVID-19 Bayesian Inference")

# Upload the dataset
uploaded_file = st.file_uploader("Upload your COVID-19 dataset (CSV file)", type="csv")

if uploaded_file is not None:
    # Read the dataset
    data = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataset to ensure it's loaded correctly
    st.write("Dataset loaded successfully:")
    st.write(data.head())
    
    # Use a subset of the data to avoid memory issues
    data_subset = data.head(100)  # Using only the first 100 rows
    
    # Define a simplified Bayesian Model
    model = BayesianModel([
        ('New cases', 'Total Confirmed cases'),
        ('New deaths', 'Death'),
        ('New recovered', 'Cured/Discharged/Migrated')
    ])
    
    # Fit the model using Maximum Likelihood Estimation
    st.write("Fitting the model...")
    model.fit(data_subset, estimator=MaximumLikelihoodEstimator)
    st.write("Model fitting complete.")
    
    # Perform inference
    st.write("Performing inference setup...")
    infer = VariableElimination(model)
    st.write("Inference setup complete.")
    
    # Set evidence values (these can be customized via Streamlit input widgets)
    new_cases = st.number_input('New cases', min_value=0, max_value=int(data['New cases'].max()), value=1)
    new_deaths = st.number_input('New deaths', min_value=0, max_value=int(data['New deaths'].max()), value=0)
    new_recovered = st.number_input('New recovered', min_value=0, max_value=int(data['New recovered'].max()), value=0)
    
    # Perform the query
    st.write("Performing the query with the provided evidence...")
    q = infer.query(variables=['Total Confirmed cases', 'Death', 'Cured/Discharged/Migrated'], evidence={
        'New cases': new_cases,
        'New deaths': new_deaths,
        'New recovered': new_recovered
    })
    
    # Extracting the results in a readable format
    # Get the indices and the values of the query result
    indices = q.variables
    values = q.values
    
    # Flatten the indices and values
    results = []
    for idx, val in enumerate(values.flatten()):
        # Decode the index combination
        index_combination = []
        current_idx = idx
        for var in reversed(indices):
            current_idx, value = divmod(current_idx, q.cardinality[q.variables.index(var)])
            index_combination.append(value)
        index_combination.reverse()
        
        # Append the combination and the probability value
        results.append(index_combination + [val])
    
    # Create a DataFrame from the results
    columns = [var for var in indices] + ['Probability']
    results_df = pd.DataFrame(results, columns=columns)
    
    # Display the results in a proper table format
    st.write("Inference results:")
    st.write(results_df)
else:
    st.write("Please upload a CSV file.")
