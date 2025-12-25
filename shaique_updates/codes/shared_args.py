# shared_args.py
import typer
from typing_extensions import Annotated

# Use Annotated to combine the Python type (e.g., str) with the Typer Option
experiment_name_arg = Annotated[str, typer.Option(help="The name for the experiment run.")]
prod_mode_arg = Annotated[bool, typer.Option(help="Run in production mode (unique folder) or dev mode (overwrite).")]
data_dir_arg = Annotated[str, typer.Option(help="Directory to store the dataset.")]
hidden_channels_arg = Annotated[int, typer.Option(help="Number of hidden channels in the GNN.")]
learning_rate_arg = Annotated[float, typer.Option(help="Learning rate for the optimizer.")]
epochs_arg = Annotated[int, typer.Option(help="Number of training epochs.")]
batch_size_arg = Annotated[int, typer.Option(help="Batch size for training and evaluation.")]
seed_arg = Annotated[int, typer.Option(help="Random seed for reproducibility.")]