import ast
import csv
import json
import os
import random

import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Create an empty graph
G = nx.Graph()

# Read nodes from nodes.csv
with open('nodes.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the first row
    for row in reader:
        node_id = row[0]
        G.add_node(node_id, name=row[1], followers=float(row[2]) if row[2] else 0,
                   popularity=float(row[3]) if row[3] else 0, genres=row[4], chart_hits=row[5])

# Read edges from edges.csv
with open('edges.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the first row
    for row in reader:
        node_id_0 = row[0]
        node_id_1 = row[1]
        G.add_edge(node_id_0, node_id_1)

# Print number of nodes and edges
print("The number of nodes are:", len(G.nodes))
print("The number of edges are:", len(G.edges))

# Find GCC
largest_cc = max(nx.connected_components(G), key=len)
GCC = G.subgraph(largest_cc).copy()

# Finding the list of followers and popularity
follower_list = list(
    GCC.nodes[node]['followers'] for node in [node for node in GCC.nodes if 'followers' in GCC.nodes[node]])
popularity_list = list(
    GCC.nodes[node]['popularity'] for node in [node for node in GCC.nodes if 'popularity' in GCC.nodes[node]])

# Find total followers and popularity
total_followers = sum(follower_list)
total_popularity = sum(popularity_list)

# Find max and min followers
max_followers = max(follower_list)
max_followers = int(max_followers)
min_followers = min(follower_list)
min_followers = int(min_followers)

# Find max and min popularity
max_popularity = max(popularity_list)
max_popularity = int(max_popularity)
min_popularity = min(popularity_list)
min_popularity = int(min_popularity)

# Find the averages
average_followers = total_followers / len(GCC.nodes)
average_followers = int(average_followers)
average_popularity = total_popularity / len(GCC.nodes)
average_popularity = int(average_popularity)

# Print GCC information
print("\nAll the following statistics are covering the GCC.")

print("The number of nodes are:", len(GCC.nodes))
print("The number of edges are:", len(GCC.edges))
print("\nThe min number of followers are:", min_followers)
print("The average number of followers are:", average_followers)
print("The max number of followers are:", max_followers)
print("\nThe min number of popularity are:", min_popularity)
print("The average popularity are:", average_popularity)
print("The max popularity are:", max_popularity)

# Find the most common genre
genre_count = {}
for node, data in GCC.nodes(data=True):
    genres = data.get('genres')
    if genres:
        genres_list = ast.literal_eval(genres)  # Convert string representation of list to actual list
        for genre in genres_list:
            genre = genre.strip().lower()
            if genre in genre_count:
                genre_count[genre] += 1
            else:
                genre_count[genre] = 1

most_common_genre = max(genre_count, key=genre_count.get)
print(f'\nThe most common genre is {most_common_genre}.')

# Find the most popular country
country_count = {}
for node, data in GCC.nodes(data=True):
    chart_hits = data.get('chart_hits')
    if chart_hits:
        hits_list = ast.literal_eval(chart_hits)
        for hit in hits_list:
            country = hit.split()[0].strip().lower()  # Extract the country code
            if country in country_count:
                country_count[country] += 1
            else:
                country_count[country] = 1

most_common_country = max(country_count, key=country_count.get)
print(f'The most popular country is {most_common_country}.')

# Finding the Degree Distribution
degrees = GCC.degree()

# Removing source node as it is not needed.
degrees = [node[1] for node in degrees]

# Finding frequency.
degree_counts = Counter(degrees)

# Finding the number of total frequencies.
total_freq = sum(degree_counts.values())

# Converting the dictionary into two lists.
degree_values = list(degree_counts.keys())
probability_values = [value / total_freq for value in degree_counts.values()]

# Graph customization code.
plt.figure(figsize=(10, 6))
plt.loglog(degree_values, probability_values, marker='o', linestyle='None')
plt.title('Degree Distribution of Giant Connected Component')
plt.xlabel('Degree')
plt.ylabel('P(k)')
plt.grid(True, which="both", ls="--")
plt.savefig("Spotify_DegreeDistribution.png")

print("\nDegree distribution graph generated.")

# Returns a dictionary values with the keys as the source node and the values as the clustering coefficients.
clustering_coefficients = nx.clustering(GCC)

# Round the clustering coefficients to 1 decimal places to limit significant figures.
clustering_coefficients = {node: round(clustering_coefficients[node], 1) for node in GCC}

# Create a dictionary of frequencies.
frequencies = {}
for value in clustering_coefficients.values():
    if value in frequencies:
        frequencies[value] += 1
    else:
        frequencies[value] = 1

# Finding the total number of frequencies.
total_freq = sum(frequencies.values())

# Find the probability of each frequency.
frequency_values = {k: v / total_freq for k, v in frequencies.items()}

# Graph customization code.
plt.figure(figsize=(10, 6))
plt.plot(frequency_values.keys(), frequency_values.values(), marker='o', linestyle='None')
plt.title('Clustering Coefficient Distribution of Giant Connected Component')
plt.xlabel('Rounded Clustering Coefficient')
plt.ylabel('C(k)')
plt.grid(True, which="both", ls="--")
plt.savefig("Spotify_ClusteringCoefficient.png")

print("Clustering coefficient distribution graph generated.\n")

# Finding the path_lengths for different percentages of the whole network.
# The all_pairs_shortest_paths is not very good at handling extremely large networks.
path_lengths = {}
sample_rates = [0.01, 0.05, 0.1]

# Check if the first txt file is missing.
if not os.path.isfile('collapsed_0.01.txt'):
    for sample_rate in sample_rates:
        sample_size = int(sample_rate * len(GCC))
        print(sample_size)

        # Sample a random subset of nodes
        sample_nodes = random.sample(list(GCC), sample_size)

        # Create a new graph with only the sampled nodes
        sample_GCC = nx.Graph()
        for node in sample_nodes:
            sample_GCC.add_node(node)
        for edge in GCC.edges():
            if edge[0] in sample_nodes and edge[1] in sample_nodes:
                sample_GCC.add_edge(edge[0], edge[1])

        path_lengths = dict(nx.all_pairs_shortest_path(sample_GCC))

        # A list of all the frequencies of the shortest paths for each source node.
        # Example. [1, 2, 3, 4, 4, 4, 4]
        # So this source node has these shortest paths to the other nodes.
        frequency_list = []

        # A dictionary for the frequency of shortest paths for all source nodes.
        # Example. {0: {1: 1. 2: 1, 3: 1, 4: 4}, ... , }
        # So this has all the frequencies of the shortest paths to all other nodes.
        frequency_dict = {}

        # Find the frequencies of the dictionaries inside path_lengths.
        for node in path_lengths:
            frequency_list = []
            for items in path_lengths[node].values():
                frequency_list.append(len(items))
            frequency_dict[node] = dict(Counter(frequency_list))

        # Write the frequency dictionary to a file
        with open('frequency_dict_{}.txt'.format(sample_rate), 'w') as f:
            json.dump(frequency_dict, f)

        # A dictionary for the collapsed frequency of all the shortest paths in the network.
        # Example. {1: 1000, 2: 10000, 3: 160000, 4: 320000, 5: 1000}
        # This simplifies all the individual dictionaries back into a smaller one.
        collapsed_frequency_dict = {}

        # Collapse the frequencies into just a few numbers.
        for node, frequencies in frequency_dict.items():
            for path_length, frequency in frequencies.items():
                if path_length in collapsed_frequency_dict:
                    collapsed_frequency_dict[path_length] += frequency
                else:
                    collapsed_frequency_dict[path_length] = frequency

        # Find the total number of frequencies
        total_freq = sum(collapsed_frequency_dict.values())

        print(collapsed_frequency_dict)

        with open('collapsed_{}.txt'.format(sample_rate), 'w') as f:
            for key, value in collapsed_frequency_dict.items():
                f.write(str(key) + ' ' + str(value) + '\n')

        # Find the probability of each frequency in the dictionary.
        collapsed_frequency_dict = {k: v / total_freq for k, v in collapsed_frequency_dict.items()}

        # Graph customization code.
        plt.figure(figsize=(10, 6))
        plt.plot(collapsed_frequency_dict.keys(), collapsed_frequency_dict.values(), marker='o', linestyle='None')
        plt.title('Shortest Path Length Distribution of Giant Connected Component')
        plt.xlabel('Distance')
        plt.ylabel('P(k)')
        plt.grid(True, which="both", ls="--")

        # Saving the graph as an image.
        plt.savefig('Spotify_SP{}Distribution.png'.format(sample_rate))
# If first file is there, assume rest are as well. Load the txt files back into objects.
else:
    for sample_rate in sample_rates:
        dict_name = 'collapsed_{}'.format(sample_rate)
        collapsed_dict = {dict_name: {}}
        with open('collapsed_{}.txt'.format(sample_rate), 'r') as f:
            for line in f:
                key, value = line.strip().split()
                collapsed_dict[int(key)] = int(value)

        # Removing the first key.
        del collapsed_dict['collapsed_{}'.format(sample_rate)]

        # Currently just saving the dictionaries. Will need to do some extrapolation to match full network.
        print(collapsed_dict)

        # Finding the diameter of the sample
        max_key = max(collapsed_dict.keys(), key=int)
        print("The diameter of this sample was:", max_key)

        # Finding the average shortest path of the sample
        sample_total = sum(key * value for key, value in collapsed_dict.items())
        sample_count = sum(collapsed_dict.values())
        sample_average = round(sample_total / sample_count, 2)
        print(f"The average shortest path of this sample was: {sample_average}\n")

        # Load the frequency dictionary from a file
        with open('frequency_dict_{}.txt'.format(sample_rate), 'r') as f:
            dict_name = 'frequency_dict_{}'.format(sample_rate)
            frequency_dict = {dict_name: json.load(f)}

# Print the average clustering of the network
print("The average clustering is:", round(nx.average_clustering(GCC), 4))

# Calculate PageRank
pr = nx.pagerank(GCC, alpha=0.85, personalization=None, max_iter=100, nstart=None, weight='weight')

# Sort the PageRank scores in descending order and get the top 50
pr_scores = sorted([(artist, pr_score) for artist, pr_score in pr.items()], key=lambda x: x[1], reverse=True)[:50]

# Open a text file and write the PageRank scores
with open('Spotify_PageRank.txt', 'w', encoding='utf-8') as f:
    for artist, pr_score in pr_scores:
        # Look up the artist name using the GCC graph
        artist_name = GCC.nodes[artist]['name']
        f.write(f"{artist_name}: {pr_score:.7f}\n")

print("\nPageRank scores written to Spotify_PageRank.txt.")

# Calculate HITS
hits = nx.hits(GCC, max_iter=100)

# Calculate the total score for each artist
total_scores = {}
for artist, hub_score in hits[0].items():
    total_scores[artist] = hub_score * hits[1][artist]

# Get the top 50 artists by total score
hits_scores = sorted([(artist, score) for artist, score in total_scores.items()], key=lambda x: x[1], reverse=True)[:50]

# Open a text file and write the top scores
with open('Spotify_HITS.txt', 'w', encoding='utf-8') as f:
    for artist, score in hits_scores:
        # Look up the artist name using the GCC graph
        artist_name = GCC.nodes[artist]['name']
        f.write(f"{artist_name}: {score:.7f}\n")

print("HITS scores written to Spotify_HITS.txt.")

# Convert the lists to sets keeping only the artist ids
pr_set = set([artist[0] for artist in pr_scores])
hits_set = set([artist[0] for artist in hits_scores])

# Find the intersection of the two sets
common_artists = pr_set & hits_set

# Save the common artists
with open('Spotify_Common_Artists_PR_HITS.txt', 'w', encoding='utf-8') as f:
    for artist in common_artists:
        # Look up the artist name using the GCC graph
        artist_name = GCC.nodes[artist]['name']
        f.write(f"{artist_name}\n")

print("Common artists from PageRank and HITS written to Spotify_Common_Artists_PR_HITS.txt.")

# Calculate the node betweenness scores
node_betweenness_scores = nx.betweenness_centrality(GCC, k=10)

# Round the node betweenness scores to four significant figures
rounded_node_betweenness_scores = {node: round(score, 8) for node, score in node_betweenness_scores.items()}

# Sort the nodes by their rounded node betweenness scores in descending order
sorted_nodes = sorted(rounded_node_betweenness_scores.items(), key=lambda x: x[1], reverse=True)

# Save the top 20 nodes to a text file
with open("Spotify_NodeBetweenness.txt", "w", encoding='utf-8') as f:
    for node, score in sorted_nodes[:20]:
        # Look up the artist name using the GCC graph
        artist_name = GCC.nodes[node]['name']
        f.write(f"{artist_name}: {score}\n")

print("\nNode betweenness scores written to Spotify_NodeBetweenness.txt.")

# Calculate the edge betweenness scores
edge_betweenness_scores = nx.edge_betweenness_centrality(GCC, k=10)

# Round the edge betweenness scores to four significant figures
rounded_edge_betweenness_scores = {edge: round(score, 8) for edge, score in edge_betweenness_scores.items()}

# Sort the edges by their rounded edge betweenness scores in descending order
sorted_edges = sorted(rounded_edge_betweenness_scores.items(), key=lambda x: x[1], reverse=True)

# Save the top 20 edges to a text file
with open("Spotify_EdgeBetweenness.txt", "w", encoding='utf-8') as f:
    for edge, score in sorted_edges[:20]:
        # Look up the artist names using the GCC graph
        artist1 = GCC.nodes[edge[0]]['name']
        artist2 = GCC.nodes[edge[1]]['name']
        f.write(f"{artist1} - {artist2}: {score}\n")

print("Edge betweenness scores written to Spotify_EdgeBetweenness.txt.")

# Using the louvain communities method to find communities.
# Girvan-Newman method was not working with such a large network.
partitions = list(nx.community.louvain_communities(GCC))

if len(partitions) == 0:
    print("The Louvain partitions algorithm didn't return any communities.")
else:
    # Get the community sizes
    community_sizes = [len(c) for c in partitions]
    # Sort the community sizes in descending order
    community_sizes.sort(reverse=True)
    # find the most famous artist in each community
    most_famous_artists = []
    most_famous_countries = []
    country_count = {}

    for community in partitions:
        max_popularity = 0
        most_famous_artist = None
        most_famous_country = None
        for artist in community:
            if 'popularity' in GCC.nodes[artist] and GCC.nodes[artist]['popularity'] > max_popularity:
                max_popularity = GCC.nodes[artist]['popularity']
                most_famous_artist = GCC.nodes[artist]['name']

            # find the most famous country in each community
            if 'chart_hits' in GCC.nodes[artist] and GCC.nodes[artist]['chart_hits'] is not None:
                charts = GCC.nodes[artist]['chart_hits']
                if len(charts) > 0:
                    hits_list = ast.literal_eval(GCC.nodes[artist]['chart_hits'])
                    for hit in hits_list:
                        country = hit.split()[0].strip().lower()  # Extract the country code
                        if country in country_count:
                            country_count[country] += 1
                        else:
                            country_count[country] = 1

        most_common_country = max(country_count, key=country_count.get)
        most_famous_countries.append(most_common_country)
        most_famous_artists.append(most_famous_artist)

        # Print the sizes of all communities
    with open('Spotify_Communities.txt', 'w') as f:
        f.write("Sizes of Top 10 communities:\n")
        for i in range(min(10, len(community_sizes))):
            f.write(
                f"Community {i + 1}: {community_sizes[i]} nodes | Most famous artist: {most_famous_artists[i]} | Most famous country: {most_famous_countries[i]}\n")
        f.write(f"\nTotal Communities: {len(community_sizes)}")
    print("\nCommunities written to Spotify_Communities.txt.")
