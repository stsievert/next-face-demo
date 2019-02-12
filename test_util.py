import graph
import train_util as train

def train_and_print_metrics(results, data, model):
    """
    Trains the model and prints the metrics
    """
    model, distances, angles, changes = train.train_model(results, data, model)
    graph.graph_metrics(distances, angles, changes)
