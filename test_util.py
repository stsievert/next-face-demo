import graph
import train_util as train

def train_and_print_metrics(results, data, model):
    """
    Trains the model and prints the metrics

    note: not used in main application, only used to help choose model
    """
    model, distances, angles, changes = train.train_model(results, data, model)
    graph.graph_metrics(distances, angles, changes)
