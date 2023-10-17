import typer
import subprocess
from pathlib import Path
import toml


from chroma_toolkit.env import env_app
from chroma_toolkit.collection import col_app
from chroma_toolkit.dataset import ds_app
from chroma_toolkit.utils.cli_utils import get_config_file

app = typer.Typer()

app.add_typer(env_app, name="env", help="Provide environment information subcommands.")
app.add_typer(col_app, name="col", help="Provide collection subcommands.")
app.add_typer(ds_app, name="ds", help="Provide dataset subcommands.")
config,config_file = get_config_file()


@app.command("choose")
def main():
    process = subprocess.Popen(["gum", "choose", f"{config}","marg"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    output, error = process.communicate(input=b"input_data\n")
    typer.echo(output.decode())



if __name__ == "__main__":
    app()