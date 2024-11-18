import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict

class HybridCommunityDetection:
    def __init__(self, graph):
        """
        Initialize the hybrid community detection algorithm.
        
        Args:
            graph (networkx.Graph): Input graph for community detection
        """
        self.graph = graph
        self.labels = {}
        self.node_importance = {}
        self.kernels = set()
        
    def identify_kernels(self, threshold=0.8):
        """
        Identify community kernels based on degree centrality.
        
        Args:
            threshold (float): Threshold for selecting high-degree nodes
        """
        # Calculate degree centrality for all nodes
        degree_cent = nx.degree_centrality(self.graph)
        sorted_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)
        
        # Select initial kernel candidates
        max_centrality = sorted_nodes[0][1]
        candidates = [node for node, cent in sorted_nodes 
                     if cent >= threshold * max_centrality]
        
        # Ensure kernels are not directly connected
        self.kernels = set()
        for node in candidates:
            if not any(nx.has_path(self.graph, node, kernel) 
                      for kernel in self.kernels):
                self.kernels.add(node)
                self.labels[node] = len(self.kernels)
                
    def calculate_node_importance(self):
        """
        Calculate node importance using degree centrality and local structure.
        """
        degree_cent = nx.degree_centrality(self.graph)
        clustering_coef = nx.clustering(self.graph)
        
        for node in self.graph.nodes():
            # Combine degree centrality with clustering coefficient
            self.node_importance[node] = (0.9 * degree_cent[node] + 
                                        0.1 * clustering_coef[node])
    
    def propagate_labels(self, max_iterations=100, threshold=0.0001):
        """
        Perform label propagation with node importance weights.
        
        Args:
            max_iterations (int): Maximum number of iterations
            threshold (float): Convergence threshold
        """
        # Initialize remaining nodes with unique labels
        current_label = len(self.kernels) + 1
        for node in self.graph.nodes():
            if node not in self.labels:
                self.labels[node] = current_label
                current_label += 1
        
        iterations = 0
        changed = True
        while changed and iterations < max_iterations:
            changed = False
            old_labels = self.labels.copy()
            
            # Randomize node order for each iteration
            nodes = list(self.graph.nodes())
            np.random.shuffle(nodes)
            
            for node in nodes:
                if node in self.kernels:
                    continue
                
                # Get neighbor labels weighted by node importance
                neighbor_labels = defaultdict(float)
                for neighbor in self.graph.neighbors(node):
                    weight = self.node_importance[neighbor]
                    neighbor_labels[old_labels[neighbor]] += weight
                
                # Select the label with maximum weighted frequency
                if neighbor_labels:
                    new_label = max(neighbor_labels.items(), 
                                  key=lambda x: x[1])[0]
                    if new_label != self.labels[node]:
                        self.labels[node] = new_label
                        changed = True
            
            iterations += 1
    
    def detect_communities(self):
        """
        Execute the complete community detection process.
        
        Returns:
            dict: Node-to-community mapping
        """
        # Step 1: Identify community kernels
        self.identify_kernels()
        
        # Step 2: Calculate node importance
        self.calculate_node_importance()
        
        # Step 3: Perform label propagation
        self.propagate_labels()
        
        return self.labels
    
    def calculate_modularity(self):
        """
        Calculate modularity score for the detected communities.
        
        Returns:
            float: Modularity score
        """
        communities = defaultdict(list)
        for node, label in self.labels.items():
            communities[label].append(node)
        
        return nx.community.modularity(self.graph, 
                                     communities.values())

def load_graph_from_csv(csv_file):
    """
    Load a graph from a CSV file containing edges.
    
    Args:
        csv_file (str): Path to the CSV file containing the graph edges
        
    Returns:
        networkx.Graph: The corresponding graph
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Create a graph from the DataFrame
    G = nx.from_pandas_edgelist(df, source='source', target=' dist')
    
    return G

def main():
    # Example usage
    csv_file = 'group_graph.csv'  # Replace with your CSV file path
    
    # Load graph from CSV
    G = load_graph_from_csv(csv_file)
    
    # Initialize and run the algorithm
    detector = HybridCommunityDetection(G)
    communities = detector.detect_communities()
    
    # Calculate and print modularity
    modularity = detector.calculate_modularity()
    print(f"Modularity Score: {modularity}")
    
    # Group nodes by community
    community_groups = defaultdict(list)
    for node, community in communities.items():
        community_groups[community].append(node)
    
    # Print communities and their members
    for community, nodes in community_groups.items():
        print(f"Community {community}: {nodes}")

if __name__ == "__main__":
    main()
