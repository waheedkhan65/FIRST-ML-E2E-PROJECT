import os

# Define the full folder and file structure
project_structure = {
    "bigmart_sales_prediction": {
        "notebooks": [
            "1_data_exploration.ipynb",
            "2_data_cleaning.ipynb",
            "3_feature_engineering.ipynb",
            "4_model_selection_automl.ipynb"
        ],
        "src": {
            "data": ["load.py", "preprocess.py"],
            "features": ["build_features.py"],
            "models": ["train_model.py", "evaluate_model.py", "predict_model.py"],
            "utils": ["logger.py", "config.py"]
        },
        "streamlit_app": ["app.py"],
        "dagshub_wandb": ["wandb_integration.py", "dagshub_integration.py"],
        "data": {
            "raw": ["train.csv", "test.csv"],
            "processed": ["train_processed.csv", "test_processed.csv"]
        },
        "outputs": {
            "models": [],
            "reports": []
        },
        "": ["requirements.txt", "README.md", ".env"]
    }
}

# Recursive function to create structure
def create_structure(base_path, structure):
    for name, content in structure.items():
        current_path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(current_path, exist_ok=True)
            create_structure(current_path, content)
        elif isinstance(content, list):
            os.makedirs(current_path, exist_ok=True)
            for file in content:
                file_path = os.path.join(current_path, file)
                with open(file_path, 'w') as f:
                    f.write(f"# {file}\n")  # add minimal placeholder
        else:
            raise ValueError(f"Unexpected content type for {name}")

# Run the function
create_structure(".", project_structure)

print("✅ Project structure created successfully!")
