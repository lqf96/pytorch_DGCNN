#! /usr/bin/env python
from __future__ import unicode_literals, print_function, division
import sys

if __name__=="__main__":
    # Show usage
    if len(sys.argv)<2:
        print("Usage: {} [Dataset name]".format(sys.argv[0]))
        exit(1)
    # Open data file
    dataset_name = sys.argv[1]
    with open("data/{}/{}.txt".format(dataset_name, dataset_name)) as f:
        # Number of graphs
        n_graphs = int(f.readline().rstrip())
        # Labels count
        labels_count = {}
        # Read label of each graph
        for _ in range(n_graphs):
            # Get number of nodes and label
            n_nodes, label = f.readline().rstrip().split(" ", 1)
            n_nodes = int(n_nodes)
            label = int(label)
            # Update label count
            count = labels_count.get(label, 0)
            count += 1
            labels_count[label] = count
            # Skip node lines
            for _ in range(n_nodes):
                f.readline()
        # Compute label frequency
        for label, label_count in labels_count.items():
            label_freq = label_count/n_graphs
            print("Label {}: amount = {}, frequency = {}".format(label, label_count, label_freq))

