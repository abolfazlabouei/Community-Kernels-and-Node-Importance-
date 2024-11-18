import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import community  # python-louvain package
from cdlib import algorithms, evaluation
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')
from main import HybridCommunityDetection

# Function to load graph from CSV
def load_graph_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    G = nx.from_pandas_edgelist(df, source='source', target=' dist')
    return G

class CommunityDetectionComparison:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.results = {}
        self.execution_times = {}
        self.modularities = {}
        
    def run_hybrid(self) -> Dict:
        start_time = time.time()
        detector = HybridCommunityDetection(self.graph)
        communities = detector.detect_communities()
        end_time = time.time()
        
        self.results['Hybrid'] = communities
        self.execution_times['Hybrid'] = end_time - start_time
        self.modularities['Hybrid'] = detector.calculate_modularity()
        return communities
    
    def run_louvain(self) -> Dict:
        start_time = time.time()
        communities = community.best_partition(self.graph)
        end_time = time.time()
        
        self.results['Louvain'] = communities
        self.execution_times['Louvain'] = end_time - start_time
        self.modularities['Louvain'] = community.modularity(communities, self.graph)
        return communities
    
    def run_leiden(self) -> Dict:
        start_time = time.time()
        leiden_communities = algorithms.leiden(self.graph)
        end_time = time.time()
        
        communities = {}
        for idx, community in enumerate(leiden_communities.communities):
            for node in community:
                communities[node] = idx
        
        self.results['Leiden'] = communities
        self.execution_times['Leiden'] = end_time - start_time
        self.modularities['Leiden'] = nx.community.modularity(self.graph, 
                                                            leiden_communities.communities)
        return communities
    
    # def run_clique_percolation(self, k: int = 3) -> Dict:
    #     """Run the Clique Percolation algorithm (alternative if needed)."""
    #     start_time = time.time()
    #     # This part would need an alternative implementation, possibly using pyclustering
    #     # clickq_communities = algorithms.clique_percolation(self.graph, k=k)
    #     end_time = time.time()
        
    #     # You would implement the clique percolation method or choose another algorithm
    #     communities = {}  # Placeholder until you implement or choose an alternative method
    #     self.results['ClickQ'] = communities
    #     self.execution_times['ClickQ'] = end_time - start_time
    #     self.modularities['ClickQ'] = 0  # Placeholder modularity value
    #     return communities

    
    def run_all_algorithms(self) -> None:
        self.run_hybrid()
        self.run_louvain()
        self.run_leiden()
        # self.run_clique_percolation()
    
    def plot_comparison(self, figsize: tuple = (15, 10)) -> None:
        fig = plt.figure(figsize=figsize)
        
        gs = plt.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        pos = nx.spring_layout(self.graph)
        
        for idx, (method, communities) in enumerate(self.results.items()):
            plt.figure(figsize=(8, 6))
            colors = [communities.get(node, 0) for node in self.graph.nodes()]
            nx.draw(self.graph, pos, node_color=colors, node_size=100,
                   cmap=plt.cm.tab20, with_labels=True)
            plt.title(f'{method} Communities')
            plt.savefig(f'{method}_communities.png')
            plt.close()
        
        methods = list(self.modularities.keys())
        values = list(self.modularities.values())
        
        sns.barplot(x=methods, y=values, ax=ax2)
        ax2.set_title('Modularity Comparison')
        ax2.set_ylabel('Modularity Score')
        ax2.tick_params(axis='x', rotation=45)
        
        methods = list(self.execution_times.keys())
        times = list(self.execution_times.values())
        
        sns.barplot(x=methods, y=times, ax=ax3)
        ax3.set_title('Execution Time Comparison')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        metrics_data = {
            'Algorithm': methods,
            'Modularity': [self.modularities[m] for m in methods],
            'Execution Time (s)': [self.execution_times[m] for m in methods],
            'Number of Communities': [len(set(comm.values())) 
                                    for comm in self.results.values()]
        }
        metrics_df = pd.DataFrame(metrics_data)
        print("\nPerformance Metrics:")
        print(metrics_df.to_string(index=False))
        
        plt.tight_layout()
        plt.savefig('community_detection_comparison.png')
        plt.close()

def main():
    # Load your graph from CSV
    graph = load_graph_from_csv('group_graph.csv')
    
    # Run comparison
    comparison = CommunityDetectionComparison(graph)
    comparison.run_all_algorithms()
    comparison.plot_comparison()
    print(f"Results saved for your graph.")

if __name__ == "__main__":
    main()
