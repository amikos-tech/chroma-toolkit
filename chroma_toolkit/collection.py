from typing_extensions import Annotated
from typing import Optional
from chromadb import Where, WhereDocument
from pydantic import SecretStr
import typer
import toml
from pathlib import Path
from chroma_toolkit.gum.gum_wrapper import GumInterface
from chroma_toolkit.telemetry import telemetry_function_decorator
from chroma_toolkit.utils.cli_utils import get_config_file, write_config_file
from rich.console import Console
from rich.table import Table
from chroma_toolkit.utils.chroma_utils import DistanceFunction, client

console = Console()
config,config_file = get_config_file()

console_interface = GumInterface

col_app = typer.Typer()

def _infer_type(v: str):
    """Infer the type of a value."""
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _add_metadata() -> dict:
    """Interactive metadata input."""
    metadata = {}
    while True:
        kv = console_interface.input("key:value NOT: Leave empty to stop", label="Metadata Item")
        if not kv:
            break
        if ":" not in kv:
            console.print("Invalid input. Please use the format 'key:value'.")
            continue
        k, v = kv.split(":")
        metadata[k] = _infer_type(v)
    return metadata

def _add_distance_function(metadata:Optional[dict], distance_function:DistanceFunction) -> dict:
    if metadata:
        metadata["hnsw:space"] = distance_function
    else:
        metadata = { "hnsw:space": distance_function }
    return metadata

@col_app.command("list")
@col_app.command("ls")
@telemetry_function_decorator
def list_collections(
    alias: Annotated[Optional[str],typer.Option(help="The alias of the environment to use.")]=None,
    extended: Annotated[Optional[bool],typer.Option(help="Show extended information.")]=False,
    ):
    """List all collections in the database."""
    if client:
        collections = client.list_collections()
        table = Table(title="Collections",*["ID","Name","Metadata"])
        if extended:
            table.add_column("Vectors")
        for collection in collections:
            if extended:
                col = client.get_collection(collection.name)
                table.add_row(str(collection.id), collection.name, str(collection.metadata),str(col.count()))
            else:
                table.add_row(str(collection.id), collection.name, str(collection.metadata))
        console.print(table)
        return collections
    else:
        typer.echo("No client found")

@col_app.command("get")
@telemetry_function_decorator
def get_collection(alias: Annotated[Optional[str],typer.Option(help="The alias of the environment to use.")]=None,
                    name: Annotated[Optional[str],typer.Option(help="The name of the collection.")]=None,
                    ):
    """Get a collection by ID."""
    if client:
        if not name:
            collections = client.list_collections()
            name=console_interface.filter([collection.name for collection in collections])
        client.get_collection(name=name)

@col_app.command("add")
@telemetry_function_decorator
def create_collection(alias: Annotated[Optional[str],typer.Option(help="The alias of the environment to use.")]=None,
                       name: Annotated[Optional[str],typer.Option(help="The name of the collection.")]=None,
                       metadata: Annotated[Optional[str],typer.Option(help="The metadata of the collection.")]=None,
                       distance_function: Annotated[DistanceFunction, typer.Option(case_sensitive=False,help="The distance function for this collection")] = None,
                       ):
    """Create a new collection."""
    if client:
        if not name:
            name = console_interface.input("Collection name")
        if not metadata:
            if console_interface.confirm(f"Do you wish to add metadata to {name}?"):
                metadata = _add_metadata()
        if not distance_function:
            if not console_interface.confirm(f"Use default distance function {DistanceFunction.L2} {name}?"):
                distance_function = console_interface.filter([distance_function.value for distance_function in DistanceFunction])
                metadata=_add_distance_function(metadata,distance_function)
        else:
            metadata=_add_distance_function(metadata,distance_function)
        col = client.create_collection(name=name,metadata=metadata)
        return col

@col_app.command("mod")
@col_app.command("modify")
@telemetry_function_decorator
def update_collection(alias: Annotated[Optional[str],typer.Option(help="The alias of the environment to use.")]=None,
                       name: Annotated[Optional[str],typer.Option(help="The name of the collection.")]=None,
                       new_name: Annotated[Optional[str],typer.Option(help="The new name of the collection.")]=None,
                       new_metadata: Annotated[Optional[str],typer.Option(help="The new metadata of the collection.")]=None,
                       merge_metadata:Annotated[Optional[bool],typer.Option(help="Merge new and old metadata. Note: New metadata keys will override old ones")]=False,):
    """Update a collection."""
    if client:
        if not name:
            collections = client.list_collections()
            name=console_interface.filter([collection.name for collection in collections])
        col = client.get_collection(name=name)
        if not new_name:
            _confirm = console_interface.confirm(f"Do you wish to change collection name?")
            if _confirm:
                _new_name = console_interface.input("New collection name")
                if console_interface.confirm(f"Change collection name from {col.name} to {_new_name}?"):
                    new_name = _new_name
        if not new_metadata:
            if console_interface.confirm(f"Do you wish to change collection metadata?"):
                _new_metadata = console_interface.input("New collection metadata")
                if console_interface.confirm(f"Change collection metadata from {col.metadata} to {new_metadata}?"):
                    new_metadata = _new_metadata if not merge_metadata else {**col.metadata,**_new_metadata}
        if new_name or new_metadata:
            col.modify(name=new_name,metadata=new_metadata)
        


@col_app.command("rm")
@col_app.command("del")
@telemetry_function_decorator
def delete_collection(alias: Annotated[Optional[str],typer.Option(help="The alias of the environment to use.")]=None,
                       name: Annotated[Optional[str],typer.Option(help="The name of the collection.")]=None,
                       ):
    """Delete a collection."""
    if client:
        if not name:
            collections = client.list_collections()
            name=console_interface.filter([collection.name for collection in collections])
        client.delete_collection(name=name)

@col_app.command("clone")
@telemetry_function_decorator
def clone_collection(alias: Annotated[Optional[str],typer.Option(help="The alias of the environment to use.")]=None,
                     source:Annotated[Optional[str],typer.Option(help="The source collection name.")]=None,
                     destination:Annotated[Optional[str],typer.Option(help="The target collection name.")]=None,
                     where:Annotated[Optional[str],typer.Option(help="Metadata filter to apply to source collection.")]=None, 
                     where_document:Annotated[Optional[str],typer.Option(help="Document filter to apply to source collection.")]=None, 
                     target_metadata:Annotated[Optional[str],typer.Option(help="Target collection metadata. If not specified source metadata will be copied.")]=None,
                     distance_function:Annotated[DistanceFunction, typer.Option(case_sensitive=False,help="The distance function for this collection")] = None,
                     ):
    """Clone a collection."""
    raise NotImplementedError