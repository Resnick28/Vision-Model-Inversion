import os
from torchview import draw_graph
import matplotlib.pyplot as plt

def visualize_model_architecture(model, model_name, input_size=(1, 1, 28, 28)):
    """
    Draw and display a neural network architecture diagram.
    
    Args:
        model: The PyTorch model to visualize
        model_name: Name of the model (used for the title)
        input_size: Input tensor size for the model
    """
    save_path = "results/classifier_model.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Generate the model graph
    graph = draw_graph(model, input_size=input_size, expand_nested=True, show_shapes=True)
    
    # Get image data without saving to disk
    img_data = graph.visual_graph.pipe(format='png')
    
    with open(save_path, "wb") as f:
        f.write(img_data)