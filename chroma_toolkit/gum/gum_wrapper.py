import subprocess
from rich import print
from pathlib import Path
from overrides import override
from chroma_toolkit import ConsoleInterface

class GumInterface(ConsoleInterface):
    @staticmethod
    @override
    def choose(values:list,label:str="Select one:",limit:int=1)->str:
        """Choose a value from a list of values."""
        process = subprocess.run(["gum", "choose","--header",f"{label}","--limit",f"{limit}", *[f"{v}" for v in values]], stdout=subprocess.PIPE, text=True)
        if process.returncode == 130:
            raise KeyboardInterrupt()
        return process.stdout.strip()

    @override
    def input(placeholder: str,label:str="Input:") -> str:
        """Get input from the user."""
        process = subprocess.run(["gum", "input","--header",f"{label}", "--placeholder", f"{placeholder}"], stdout=subprocess.PIPE, text=True)
        if process.returncode == 130:
            raise KeyboardInterrupt()
        return process.stdout.strip().strip()

    @staticmethod
    @override
    def file(start_dir:Path = Path.cwd(),label:str="Select a file:")->str:
        """Get a file from the user."""
        print(label)
        process = subprocess.run(["gum", "file",f"{start_dir}"], stdout=subprocess.PIPE, text=True)
        if process.returncode == 130:
            raise KeyboardInterrupt()
        return process.stdout.strip()

    @staticmethod
    @override
    def directory(start_dir:Path = Path.cwd(),label:str="Select a directory:")->str:
        """Get a directory from the user."""
        process = subprocess.run(["gum", "file","--directory", f"{start_dir}"], stdout=subprocess.PIPE, text=True)
        if process.returncode == 130:
            raise KeyboardInterrupt
        return process.stdout.strip()

    @override
    def password(placeholder: str,label:str="Password:") -> str:
        """Get a password from the user."""
        process = subprocess.run(["gum","input","--header",f"{label}", "--password", "--placeholder", f"{placeholder}"], stdout=subprocess.PIPE, text=True)
        if process.returncode == 130:
            raise KeyboardInterrupt
        return process.stdout.strip()

    @staticmethod
    @override
    def confirm(message:str)->bool:
        """Get a confirmation from the user."""
        process = subprocess.run(["gum", "confirm", f"{message}"], stdout=subprocess.PIPE, text=True)
        if process.returncode == 130:
            raise KeyboardInterrupt
        return process.returncode == 0

    @staticmethod
    @override
    def filter(items:list, label:str="Select:",limit:int=1)->str:
        """Get a list of filters from the user."""
        echo_process = subprocess.Popen(["echo", "\n".join([f"{v}" for v in items])], stdout=subprocess.PIPE)
        gum_process = subprocess.Popen(["gum", "filter", "--header", f"{label}","--limit",f"{limit}"], stdin=echo_process.stdout, stdout=subprocess.PIPE, text=True)
        output, error = gum_process.communicate()
        if gum_process.returncode == 130:
            raise KeyboardInterrupt
        return output.strip()