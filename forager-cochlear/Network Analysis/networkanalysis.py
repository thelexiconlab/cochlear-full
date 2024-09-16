import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

nodes = pd.read_csv(r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\forager-cochlear\nodes.csv")
edges = pd.read_csv(r"C:\Users\c.hambric\OneDrive - Bowdoin College\Documents\cochlear-full\forager-cochlear\valid_edge_list.csv")
edges_list = [(row.Word1, row.Word2) for row in edges.itertuples(index=False)]

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges_list)


pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=500, font_size=12, font_weight='bold')
plt.show()
connected_components = nx.connected_components(G)
for component_nodes in connected_components:
    # Create subgraph for each component
    component_subgraph = G.subgraph(component_nodes)
    
    if len(component_subgraph.nodes()) > 1:  # Ensure there's more than one node to compute metrics
        print(f"Number of nodes: {len(component_subgraph.nodes())}")
        print(f"Number of edges: {len(component_subgraph.edges())}")
        average_clustering = nx.average_clustering(component_subgraph)
        print(f"Average Clustering Coefficient: {average_clustering}")
        density = nx.density(component_subgraph)
        print(f"Graph Density: {density}")
        try:
            shortest_path = nx.average_shortest_path_length(component_subgraph)
            print(f"Average Shortest Path: {shortest_path}")
        except nx.NetworkXError as e:
            print(f"Error calculating shortest path: {e}")
    else:
        print("Skipping nodes with no connections.")