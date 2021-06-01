import networkx as nx
import os
from pathlib import Path
from matplotlib import pyplot as plt
from texttable import Texttable
from tqdm import tqdm
import pandas as pd
from create_directed_atlas import create_directed_atlas

import argparse

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]]+[[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def parameter_parser():
    """
    Calculating counts of orbital roles in connected graphlets.
    Representations are sorted by ID.
    """
    parser = argparse.ArgumentParser(description="Extracting the features.")

    parser.add_argument('--graph-input',
                        nargs='?',
                        default="./input/whatever",
                        help='Edge list csv path.')

    parser.add_argument('--output',
                        nargs='?',
                        default='./output/arsenal_3749552_orbital_features_1.csv',
                        help='Feature output path.')

    parser.add_argument('--graphlet-size',
                        type=int,
                        default=3,
                        help='Maximal graphlet size. Default is 3.')

    return parser.parse_args()

class MotifCounterMachine(object):
    """
    Connected motif orbital role counter.
    """
    def __init__(self, graph, args):
        """
        Creating an orbital role counter machine.
        :param graph: NetworkX graph.
        :param args: Arguments object.
        """
        self.graph = graph
        self.args = args

    def create_edge_subsets(self):
        """
        Enumerating connected subgraphs with size 2 up to the graphlet size.
        """
        print("\nEnumerating subgraphs.\n")
        self.edge_subsets = dict()
        subsets = [[edge[0], edge[1]] for edge in self.graph.edges()]
        self.edge_subsets[2] = subsets
        unique_subsets = dict()
        for i in range(3, self.args.graphlet_size+1):
            print("Enumerating graphlets with size: " +str(i) + ".")
            for subset in tqdm(subsets):
                for node in subset:
                    for neb in self.graph.neighbors(node):
                        new_subset = subset+[neb]
                        if len(set(new_subset)) == i:
                            new_subset.sort()
                            unique_subsets[tuple(new_subset)] = 1
            subsets = [list(k) for k, v in unique_subsets.items()]
            self.edge_subsets[i] = subsets
            unique_subsets = dict()

    def enumerate_graphs(self):
        """
        Creating a hash table of the benchmark motifs.
        """
        graphs = create_directed_atlas()
        self.interesting_graphs = {i: [] for i in range(1, self.args.graphlet_size+1)}
        for graph in graphs:
            if graph.number_of_nodes() > 0 and graph.number_of_nodes() < self.args.graphlet_size+1:
                    self.interesting_graphs[graph.number_of_nodes()].append(graph)

    def enumerate_categories(self):
        """
        Creating a hash table of benchmark orbital roles.
        TODO: Check if setting degree here works OK, since we're now working with undirected grpahs.
        """
        main_index = 0
        self.categories = dict()
        for size, graphs in self.interesting_graphs.items():
            self.categories[size] = dict()
            for index, graph in enumerate(graphs):
                self.categories[size][index] = dict()
                degrees = list(set([(graph.out_degree(node), graph.in_degree(node)) for node in graph.nodes()]))
                for degree in degrees:
                    self.categories[size][index][degree] = main_index
                    main_index = main_index + 1
        self.unique_motif_count = main_index + 1

    def setup_features(self):
        """
        Counting all the orbital roles.
        """
        print("\nCounting orbital roles.\n")
        self.features = {node: {i:0 for i in range(self.unique_motif_count)}for node in self.graph.nodes()}
        for size, node_lists in self.edge_subsets.items():
            graphs = self.interesting_graphs[size]
            for nodes in tqdm(node_lists):
                sub_gr = self.graph.subgraph(nodes)

                for index, graph in enumerate(graphs):
                    if nx.is_isomorphic(sub_gr, graph):
                        for node in sub_gr.nodes():
                            self.features[node][self.categories[size][index][(sub_gr.out_degree(node), sub_gr.in_degree(node))]] += 1
                        break

    def create_tabular_motifs(self):
        """
        Creating a table with the orbital role features.
        """
        print("Saving the dataset.")
        self.binned_features = {node: [] for node in self.graph.nodes()}
        self.motifs = [[n]+[self.features[n][i] for i in  range(self.unique_motif_count)] for n in self.graph.nodes()]
        self.motifs = pd.DataFrame(self.motifs)
        self.motifs.columns = ["id"] + ["role_"+str(index) for index in range(self.unique_motif_count)]
        self.motifs.to_csv(self.args.output, index=None)

    def write_dictionary(self):
        with open('./output/dictionary.txt', 'w') as f:
            counter = 0
            for node_count in self.categories:
                for graphlet in self.categories[node_count]:
                    for degree in self.categories[node_count][graphlet]:
                        f.write(f'Orbit {counter}: {node_count}-node graphlet number {graphlet+1} with (out, in) degrees {degree}\n')
                        counter += 1
    def extract_features(self):
        """
        Executing steps for feature extraction.
        """
        self.create_edge_subsets()
        self.enumerate_graphs()
        self.enumerate_categories()
        self.setup_features()
        self.write_dictionary()
        self.create_tabular_motifs()





if __name__ == "__main__":

    p=Path(os.getcwd())

    nets = p.parent.parent

    #G2 = nx.read_pajek(str(nets) + '\\nets\\7430\\7430_period_1.net')

    args = parameter_parser()
    tab_printer(args)
    graph = nx.DiGraph(nx.read_pajek(str(nets) + '\\nets\\3749552\\3749552_Arsenal (1)_period_1.net'))
    model = MotifCounterMachine(graph, args)
    model.extract_features()

