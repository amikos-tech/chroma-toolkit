from enum import Enum
import chromadb
from chromadb.config import Settings

from chroma_toolkit.utils.cli_utils import get_config_file, get_default_env

def get_chroma_client(env_config:dict):
    if env_config['env_type'] == "Local":
        chroma_client = chromadb.PersistentClient(path=env_config['path'])
        return chroma_client
    else:
        settings = Settings()
        if env_config['auth_type']!= "None":
            if env_config['auth_type'] == "Token":
                settings=Settings(chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                      chroma_client_auth_credentials=env_config['credentials'])
            elif env_config['auth_type'] == "Basic":
                settings=Settings(chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",
                                  chroma_client_auth_credentials=env_config['credentials'])
        chroma_client = chromadb.HttpClient(host=env_config['path'],settings=settings)
        return chroma_client

config,config_file = get_config_file()

default_env=get_default_env(config)
client:chromadb.Client = None
if default_env:
    client = get_chroma_client(default_env)


class DistanceFunction(str, Enum):
    L2 = "l2"
    COS = "cosine"
    IP = "ip"
