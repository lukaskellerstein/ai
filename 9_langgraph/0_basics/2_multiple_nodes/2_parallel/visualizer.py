import io
from PIL import Image

# ---------------------------
# Visualize the graph
# ---------------------------

def visualize(graph, output_file_name):
    try:
        png = graph.get_graph().draw_mermaid_png()
        # Create a Pillow Image object from the image data
        pil_image = Image.open(io.BytesIO(png))  # Replace io.BytesIO with appropriate stream if necessary

        # Save the image to a file
        output_file = output_file_name  # Replace with your desired output file path
        pil_image.save(output_file, 'PNG')
    except Exception:
        # This requires some extra dependencies and is optional
        pass
    
