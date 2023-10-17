from abc import ABC, abstractmethod
import importlib
import inspect
from pathlib import Path
from typing import Type, TypeVar, cast
from overrides import EnforceOverrides

class ConsoleInterface(ABC, EnforceOverrides):
    @staticmethod
    @abstractmethod
    def choose(values: list, label: str = "Select one:",limit:int=1) -> str:
        pass

    @staticmethod
    @abstractmethod
    def input(placeholder: str, label: str = "Input:") -> str:
        pass

    @staticmethod
    @abstractmethod
    def file(start_dir: Path = Path.cwd(), label: str = "Select a file:") -> str:
        pass

    @staticmethod
    @abstractmethod
    def directory(start_dir: Path = Path.cwd(), label: str = "Select a directory:") -> str:
        pass

    @staticmethod
    @abstractmethod
    def password(placeholder: str, label: str = "Password:") -> str:
        pass

    @staticmethod
    @abstractmethod
    def confirm(message: str) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def filter(items: list, label: str = "Select:",limit:int=1) -> str:
        pass


C = TypeVar("C")


def get_class(fqn: str, type: Type[C]) -> Type[C]:
    """Given a fully qualifed class name, import the module and return the class"""
    module_name, class_name = fqn.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cast(Type[C], cls)


def get_fqn(cls: Type[object]) -> str:
    """Given a class, return its fully qualified name"""
    return f"{cls.__module__}.{cls.__name__}"