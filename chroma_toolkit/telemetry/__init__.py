
from typing import Any, Callable, Type, Dict, Tuple
from posthog import Posthog
from pathlib import Path
import uuid
from functools import wraps

import time

from chroma_toolkit.utils.cli_utils import get_config_file, write_config_file

config,config_file = get_config_file()
telemetry_config = None

if "telemetry" in config.keys():
    telemetry_config:dict = config["telemetry"]
    if "enabled" not in telemetry_config.keys():
        telemetry_config["enabled"] = True
        write_config_file(config,config_file)

else:
    telemetry_config = {"enabled":True}
    config["telemetry"] = telemetry_config
    write_config_file(config,config_file)

posthog = Posthog(project_api_key='phc_P9u0LW6NWbQ24RtVySH18294iJa9pYXk7K4fwbFTgMG', host='https://eu.posthog.com')

def create_or_get_telemetry_user_id() -> str:
    Path.home().joinpath(".chroma-toolkit").mkdir(exist_ok=True)
    Path.home().joinpath(".chroma-toolkit").joinpath("telemetry_user_id").touch(exist_ok=True)
    telemetry_file = Path.home().joinpath(".chroma-toolkit").joinpath("telemetry_user_id").absolute()
    with open(telemetry_file, "r") as f:
        telemetry_user_id = f.read()
        if not telemetry_user_id:
            telemetry_user_id = str(uuid.uuid4())
            with open(telemetry_file, "w") as f:
                f.write(telemetry_user_id)
    return telemetry_user_id

telemetry_user_id = create_or_get_telemetry_user_id()

def telemetry_class_decorator() -> Callable[[type], Type[Any]]:
    def _decorator(cls: type) -> Type[Any]:
        methods = [method for method in dir(cls) if callable(getattr(cls, method)) and not method.startswith("_")]
        for target_method in methods:
            original_method = getattr(cls, target_method)

            def wrapped(self: Any, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) \
                    -> Any:
                posthog.capture(cls.__name__, target_method)
                return original_method(self, *args, **kwargs)

            setattr(cls, target_method, wrapped)
        return cls
    return _decorator

def telemetry_function_decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            res= func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            if telemetry_config["enabled"]:
                posthog.capture(telemetry_user_id,func.__name__,properties={"args":args,"kwargs":kwargs, "execution_time": execution_time,"library":"chroma-toolkit"})
            return res
        except KeyboardInterrupt as e:
            end_time = time.time()
            execution_time = end_time - start_time
            if telemetry_config["enabled"]:
                posthog.capture(telemetry_user_id,f"{func.__name__}",properties={"args":args,"kwargs":kwargs,"interrupted":True,"execution_time": execution_time,"library":"chroma-toolkit"})
            raise e
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            if telemetry_config["enabled"]:
                posthog.capture(telemetry_user_id,f"{func.__name__}",properties={"args":args,"kwargs":kwargs,"error":str(e),"execution_time": execution_time,"library":"chroma-toolkit"})
            raise e
    return wrapper
