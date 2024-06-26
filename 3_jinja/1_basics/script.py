from jinja2 import Environment, FileSystemLoader
import os

# Define the user object
payload = {
    "name": "John Doe",
    "role": "admin",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Summarize the text delimited by triple backticks into a single sentence.",
        },
    ],
}

# Set up the Jinja2 environment
template_loader = FileSystemLoader(searchpath=os.path.dirname(__file__))
env = Environment(loader=template_loader, trim_blocks=True, lstrip_blocks=True)

# Load the template
template = env.get_template("my-template.jinja2")

# Render the template with the user object
rendered_template = template.render(payload=payload).rstrip()

# Print or save the rendered template
print(rendered_template)

# Optionally, you can save the rendered template to an HTML file
with open("my-template-rendered.txt", "w") as f:
    f.write(rendered_template)
