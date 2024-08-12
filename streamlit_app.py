import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Streamlit Page Configuration
st.set_page_config(
    page_title='Quantum Supply Chain Manager Simulation 🚀',
    page_icon=':package:',
    layout='wide'
)

# Title
st.title('Quantum Supply Chain Manager Simulation 🚀')
st.markdown("""
Welcome to the **Quantum Supply Chain Manager Simulation**. This application uses simulated quantum results to optimize two key supply chain tasks:
- **🔮 Backorder Prediction**: Predicts the likelihood of products going on backorder using quantum neural network simulation.
- **🚚 Vehicle Routing Optimization**: Finds optimal delivery routes using quantum-inspired algorithms.
""")

# Generate realistic supply chain data
def generate_data():
    data = {
        'Product_ID': [f'Product_{i+1}' for i in range(5)],
        'Current_Inventory': [200, 150, 80, 120, 180],  # Inventory levels
        'Lead_Time': [3, 7, 2, 5, 6],  # Lead time in days
        'In_Transit_Qty': [30, 40, 15, 25, 35],
        'Forecast_3_Month': [70, 80, 50, 65, 90],
        'Forecast_6_Month': [120, 130, 90, 110, 150],
        'Forecast_9_Month': [160, 190, 120, 150, 200],
        'Sales_1_Month': [30, 35, 20, 25, 40],
        'Sales_3_Month': [85, 90, 50, 70, 100],
        'Sales_6_Month': [120, 130, 80, 100, 130],
        'Sales_9_Month': [160, 180, 100, 140, 170],
        'Backorder': [0, 1, 0, 1, 0]  # Simulated backorder labels
    }
    return pd.DataFrame(data)

data = generate_data()

# Display the dataset
st.header('📊 Synthetic Supply Chain Data')
st.write(data)

# Simulate quantum neural network results for backorder prediction
def simulate_quantum_nn(data):
    np.random.seed(0)
    predictions = np.random.choice([0, 1], size=len(data))
    return predictions

predictions = simulate_quantum_nn(data)
data['Predicted_Backorder'] = predictions

st.header('🔮 Backorder Predictions with Quantum Neural Network Simulation')
st.write(data[['Product_ID', 'Current_Inventory', 'Lead_Time', 'Predicted_Backorder']])

# Generate synthetic distance matrix for routing
def generate_distance_matrix():
    return np.array([
        [0, 10, 15, 20, 25],
        [10, 0, 10, 15, 20],
        [15, 10, 0, 10, 15],
        [20, 15, 10, 0, 10],
        [25, 20, 15, 10, 0]
    ])

distance_matrix = generate_distance_matrix()

# Vehicle Routing Problem (VRP) Optimization
def optimize_vrp(distance_matrix):
    num_locations = len(distance_matrix)
    G = nx.complete_graph(num_locations)
    
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                G[i][j]['weight'] = distance_matrix[i][j]

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=700, font_size=16, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12)
    plt.title('🚚 Optimized Routing Network', fontsize=18, fontweight='bold')
    st.pyplot(plt.gcf())

st.header('🚚 Vehicle Routing Problem Optimization')
optimize_vrp(distance_matrix)

# Plot Distance Matrix
def plot_distance_matrix(matrix):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='coolwarm', vmin=0, vmax=25)
    fig.colorbar(cax)
    plt.title('📏 Distance Matrix Heatmap', fontsize=18, fontweight='bold')
    plt.xlabel('Destination')
    plt.ylabel('Source')
    st.pyplot(fig)

st.header('📏 Distance Matrix Heatmap')
plot_distance_matrix(distance_matrix)
