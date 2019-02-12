import graph
import train

def train_and_print_metrics(results, data, model):
    """
    This is here just to make it simpler when training and instantly printing
    the results
    """
    model, distances, angles, changes = train.train_model(results, data, model)
    graph.graphMetrics(distances, angles, changes)
