from pydantic import SecretStr
import typer
import toml
from pathlib import Path
from chroma_toolkit.gum.gum_wrapper import GumInterface
from chroma_toolkit.telemetry import telemetry_function_decorator
from chroma_toolkit.utils.cli_utils import get_config_file, write_config_file
from rich.console import Console
from rich.table import Table

console = Console()
config,config_file = get_config_file()

console_interface = GumInterface

env_app = typer.Typer()

def _set_default_env(alias: str):
    for env in config.keys():
        if env == alias:
            config[env]["default"] = True
        else:
            config[env]["default"] = False

@env_app.command("add")
@telemetry_function_decorator
def add_environment(
    alias = typer.Option(default=None, help="The alias of the environment."),
                     path= typer.Option(default=None, help="The path to the environment."),
                     auth_type = typer.Option(default=None, help="The authentication type."),
                     credentials = typer.Option(default=None, help="The credentials."),
                     description = typer.Option(default=None, help="The description of the environment."),
                     set_default = typer.Option(default=False, help="Set this environment as the default environment.")):
    """Add a new environment."""
    _alias = alias
    if not alias:
        _alias= console_interface.input("Environment alias")
    _path = path
    _env_type = None
    if not path:
        _env_type = console_interface.choose(["Local","Remote"])
        typer.echo(_env_type == "Local")
        if _env_type == "Local":
            _path = console_interface.directory()
        else:
            _path = console_interface.input("http://localhost:8000")
    _auth_type = auth_type
    _credentials:SecretStr = SecretStr(credentials)
    if not auth_type:
        _auth_type = console_interface.choose(["None","Basic","Token"],label="Authentication type")
        if _auth_type == "Basic":
            _user = console_interface.input("username",label="Username:")
            _password = console_interface.password("password",label="Password:")
            _credentials = SecretStr(f"{_user}:{_password}")
        elif _auth_type == "Token":
            _credentials = SecretStr(console_interface.password("token",label="Token:"))
    if not description:
        _description = console_interface.input("Describe the environment",label="Environment Description:")
    
    _set_default=set_default
    if not _set_default:
        _set_default = console_interface.confirm(f"Set {_alias} environment as the default environment?")
        
            
    table = Table("Property", "Value")
    table.add_row("Alias", f"{_alias}")
    table.add_row("Environment Type", f"{_env_type}")
    table.add_row("Path", f"{_path}")
    if _auth_type!="None":
        table.add_row("Auth Type", f"{_auth_type}")
        table.add_row("Credentials", f"{_credentials}")
    table.add_row("Description", f"{_description}")
    table.add_row("Default", f"{_set_default}")
    console.print(table)

    _confirmed = console_interface.confirm("Is this correct?")
    if _confirmed:
        config[f"env:{_alias}"] = {}
        config[f"env:{_alias}"]["alias"] = _alias
        config[f"env:{_alias}"]["env_type"] = _env_type
        config[f"env:{_alias}"]["path"] = _path
        config[f"env:{_alias}"]["auth_type"] = _auth_type
        config[f"env:{_alias}"]["credentials"] = _credentials.get_secret_value()
        config[f"env:{_alias}"]["description"] = _description
        config[f"env:{_alias}"]["default"] = _set_default
        if _set_default:
            _set_default_env(f"env:{_alias}")
        write_config_file(config,config_file)


@env_app.command("rm")
@telemetry_function_decorator
def remove_environment(alias = typer.Option(default=None, help="The alias of the environment."),):
    """Remove an environment."""
    _alias = alias
    if not alias:
        _alias=console_interface.filter([f"{env.lstrip('env:')}" for env in config.keys()],label="Select an environment to remove:")
    _confirmed = console_interface.confirm(f"Are you sure you want to delete {_alias}?")
    if _confirmed:
        typer.echo("Removing environment...")
        del config[f"env:{_alias}"]
        write_config_file(config,config_file)


@env_app.command("ls")
@telemetry_function_decorator
def list_environments():
    """List all environments."""
    table = Table("Alias", "Path", "Auth Type", "Description", "Default")
    for env in config.keys():
        if env.startswith("env:"):
            table.add_row(f"{env.lstrip('env:')}", f"{config[env]['path']}", f"{config[env]['auth_type']}", f"{config[env]['description']}", f"{config[env]['default']}")
    console.print(table)

@env_app.command("default")
@telemetry_function_decorator
def set_default_environment(alias = typer.Option(default=None, help="The alias of the environment."),):
    """Set the default environment."""
    _alias = alias
    if not alias:
        _alias=console_interface.filter(items=[f"{env.lstrip('env:')}" for env in config.keys()],label="Select an environment to set as default:")
    _confirmed = console_interface.confirm(f"Are you sure you want to set {_alias} as the default environment?")
    if _confirmed:
        _set_default_env(f"env:{_alias}")
        write_config_file(config,config_file)