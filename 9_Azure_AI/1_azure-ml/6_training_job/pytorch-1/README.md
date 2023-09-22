0. Training environment (Custom) created via env.yaml in AML

1. Training data are uploaded in AML Data section

2. Training of the model is done via Jobs in AML

Command: `python main.py --data ${{inputs.train_data}} --output_dir ${{outputs.output_folder}}`
