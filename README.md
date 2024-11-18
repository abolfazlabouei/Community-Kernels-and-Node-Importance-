# Community-Kernels-and-Node-Importance-
This repository contains the implementation of a novel method for community detection in social networks, particularly applied to Telegram groups. The algorithm combines structural data (shared users between groups) and textual content (group names/descriptions) to improve modularity and address the instability of Label Propagation Algorithm (LPA).

## Features
Implements LPA-based community detection with content fusion.
Constructs a weighted graph where:
Nodes represent Telegram groups.
Edges represent shared members between groups, with weights based on user overlap and content similarity.
Supports multiple algorithms, including Louvain, Leiden, and LPA.
Uses FastText for extracting content embeddings to enhance edge weights.
Outputs results with modularity scores for evaluation.