
from enum import Enum
from typing import Optional, Sequence
import uuid
from rich.progress import track
from typing_extensions import Annotated
from chromadb import Where, WhereDocument
import typer
import shutil
import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
import toml
import itertools
from pathlib import Path
from chroma_toolkit.gum.gum_wrapper import GumInterface
from chroma_toolkit.telemetry import telemetry_function_decorator
from chroma_toolkit.utils.cli_utils import get_config_file, write_config_file
from rich.console import Console
from rich.table import Table
from chroma_toolkit.utils.chroma_utils import DistanceFunction, client
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HfFolder
from concurrent.futures import ThreadPoolExecutor
console = Console()
config, config_file = get_config_file()

console_interface = GumInterface

ds_app = typer.Typer()


class DatasourceType(str, Enum):
    HF = "huggingface"
    LOCAL = "local"


def _create_import_state(import_id: str, alias: str, dataset: str,
                         dataset_type: DatasourceType,
                         dataset_stream: bool,
                         collection: str,
                         create_collection: bool,
                         import_batch_size: int,
                         limit: int,
                         document_feature: str,
                         embedding_feature: str,
                         metadata_features: list
                         ):
    Path.home().joinpath(".chroma-toolkit").joinpath("imports").mkdir(exist_ok=True)
    Path.home().joinpath(".chroma-toolkit").joinpath("imports").joinpath(
        f"import_{import_id}.toml").touch(exist_ok=True)
    _import_file = Path.home().joinpath(
        ".chroma-toolkit").joinpath("imports").joinpath(f"import_{import_id}.toml").absolute()
    import_config = {
        "import_id": import_id,
        "alias": alias,
        "dataset": dataset,
        "dataset_type": dataset_type.value,
        "dataset_stream": dataset_stream,
        "collection": collection,
        "create_collection": create_collection,
        "import_batch_size": import_batch_size,
        "limit": limit,
        "samples_imported": 0,
        "batch_imported": 0,
        "document_feature": document_feature,
        "embedding_feature": embedding_feature,
        "metadata_features": metadata_features,
    }
    with open(_import_file, "w") as f:
        toml.dump(import_config, f)
    return import_config, _import_file


def _load_import_state(import_state_file: str):
    with open(import_state_file, "r") as f:
        return toml.load(f), import_state_file


def _persist_import_state(import_state_file: str, import_config: dict):
    with open(import_state_file, "w") as f:
        toml.dump(import_config, f)


def _create_export_state(export_id: str, **kwargs):
    Path.home().joinpath(".chroma-toolkit").joinpath("exports").mkdir(exist_ok=True)
    Path.home().joinpath(".chroma-toolkit").joinpath("exports").joinpath(
        f"export_{export_id}.toml").touch(exist_ok=True)
    _export_file = Path.home().joinpath(
        ".chroma-toolkit").joinpath("exports").joinpath(f"export_{export_id}.toml").absolute()
    export_config = {
        "export_id": export_id,
        **kwargs
    }
    with open(_export_file, "w") as f:
        toml.dump(export_config, f)
    return export_config, _export_file


def _load_export_state(export_state_file: str):
    with open(export_state_file, "r") as f:
        return toml.load(f), export_state_file


def _persist_export_state(export_state_file: str, export_config: dict):
    with open(export_state_file, "w") as f:
        toml.dump(export_config, f)


def _infer_hf_type(value):
    """
    Infers the Hugging Face data type from a Python type.

    Args:
        value: The value to infer the data type from.

    Returns:
        The inferred Hugging Face data type.
    """
    if isinstance(value, bool):
        return datasets.Value("bool")
    elif isinstance(value, int):
        return datasets.Value("int32")
    elif isinstance(value, float):
        return datasets.Value("float32")
    elif isinstance(value, str):
        return datasets.Value("string")
    elif isinstance(value, list):
        if all(isinstance(elem, int) for elem in value):
            return datasets.features.Sequence(feature=datasets.Value(dtype='int32'))
        elif all(isinstance(elem, float) for elem in value):
            return datasets.features.Sequence(feature=datasets.Value(dtype='float32'))
        elif all(isinstance(elem, str) for elem in value):
            return datasets.features.Sequence(feature=datasets.Value(dtype='string'))
        else:
            raise ValueError("Unsupported list type")
    else:
        raise ValueError("Unsupported type")


def get_first_x(iterable, x):
    return list(itertools.islice(iterable, x))


@ds_app.command("import")
@ds_app.command("i")
@telemetry_function_decorator
def import_dataset(
        alias: Annotated[Optional[str], typer.Option(
            help="The alias of the environment to use.")] = None,
        type: Annotated[DatasourceType, typer.Option(
            help="The alias of the environment to use.")] = DatasourceType.HF,
        dataset: Annotated[Optional[str], typer.Option(
            help="The dataset it. For HF this is the id, for local it is the path.")] = "KShivendu/dbpedia-entities-openai-1M",
        dataset_stream: Annotated[Optional[bool], typer.Option(
            help="Stream the dataset. Progress will not be shown but reduced memory requirements.")] = False,
        collection: Annotated[Optional[str], typer.Option(
            help="Collection name to import the data in.")] = None,
        create_collection: Annotated[Optional[bool], typer.Option(
            help="Create a new collection if it doesn't exist.")] = False,
        import_batch_size: Annotated[Optional[int], typer.Option(
            help="The size of a batch to import.")] = 100,
        limit: Annotated[Optional[int], typer.Option(
            help="The limit of records to import.")] = -1,
        resume_import_file: Annotated[Optional[str], typer.Option(
            help="The file to resume import from.")] = None,
):
    if client:
        _is_resume = False
        if resume_import_file:
            _import_config, _import_state_file = _load_import_state(
                resume_import_file)
            _import_id = _import_config["import_id"]
            collection = _import_config["collection"]
            limit = _import_config["limit"]
            import_batch_size = _import_config["import_batch_size"]
            create_collection = _import_config["create_collection"]
            dataset_stream = _import_config["dataset_stream"]
            dataset = _import_config["dataset"]
            type = DatasourceType(_import_config["dataset_type"])
            document_feature = _import_config["document_feature"]
            embedding_feature = _import_config["embedding_feature"]
            metadata_features = _import_config["metadata_features"]
            _is_resume = True
        else:
            document_feature = None
            embedding_feature = None
            metadata_features = None
            _import_id = str(uuid.uuid4())
        col = None
        if not collection:
            collections = client.list_collections()
            if not create_collection and all([collection.name != collection for collection in collections]):
                collection = console_interface.filter(
                    [collection.name for collection in collections])
                col = client.get_collection(collection)
            else:
                col = client.create_collection(collection)
        else:
            col = client.get_collection(collection)
        if type == DatasourceType.HF:
            if not dataset:
                dataset = console_interface.input("Dataset ID")
            _dataset = load_dataset(
                dataset, split='train', streaming=dataset_stream)
            _dataset_len = _dataset.num_rows if hasattr(
                _dataset, "num_rows") else -1
            limit = limit if limit != -1 else _dataset_len
            features = _dataset.features
            if not document_feature:
                document_feature = console_interface.choose(
                    [feature for feature in features.keys()], label="Document Feature")
            if not embedding_feature:
                embedding_feature = console_interface.choose(
                    [feature for feature in features.keys()]+["None"], label="Embedding Feature")
                if embedding_feature == "None":
                    console.print(
                        "No embedding feature selected. Embeddings will be generated using default EF.", style="bold orange1")
            if not metadata_features:
                remaining_features = [feature for feature in features.keys(
                ) if feature != document_feature and feature != embedding_feature]
                metadata_features = console_interface.choose(
                    remaining_features, label="Metadata Features From Dataset", limit=len(remaining_features))
                metadata_features = metadata_features.replace(
                    "\n", ",").split(",")
            table = Table(show_header=True,
                          header_style="bold magenta", title="Import Details")
            table.add_column("Collection")
            table.add_column("Document Feature")
            table.add_column("Embedding Feature")
            table.add_column("Metadata Features")
            table.add_column("Import Samples")
            table.add_column("Batch Size")
            table.add_column("Total Samples in Dataset")
            table.add_row(collection, document_feature, embedding_feature, ",".join(metadata_features), str(
                limit) if limit != -1 else "all", str(import_batch_size), str(_dataset_len))
            console.print(table)
            if not console_interface.confirm("Do you want to continue?"):
                return
            if not _is_resume:
                _import_config, _import_state_file = _create_import_state(_import_id,
                                                                          alias,
                                                                          dataset,
                                                                          type,
                                                                          dataset_stream,
                                                                          collection,
                                                                          create_collection,
                                                                          import_batch_size,
                                                                          limit,
                                                                          document_feature,
                                                                          embedding_feature,
                                                                          metadata_features)
            console.print(
                f"Import state file: {_import_state_file}", style="bold green")
            _batch = {
                "documents": [],
                "embeddings": [],
                "metadatas": [],
                "ids": [],
            }

            def add_to_col(batch):
                col.add(**batch)

            # for i, example in enumerate(_dataset):
            if limit == -1:
                console.print(
                    "Streaming import of dataset. Progress bar unavailable.", style="bold orange1")
                for j, example in enumerate(_dataset):
                    if _is_resume and j < _import_config["samples_imported"]:
                        continue
                    _batch["documents"].append(example[document_feature])
                    if embedding_feature != "None":
                        _batch["embeddings"].append(example[embedding_feature])
                    else:
                        del _batch["embeddings"]
                    _batch["ids"].append(str(uuid.uuid4()))
                    if len(metadata_features) > 0:
                        _batch["metadatas"].append(
                            {f"{feature}": f"{example[feature]}" for feature in metadata_features})
                    else:
                        _batch["metadatas"].append(None)
                    if len(_batch["ids"]) == import_batch_size:
                        col.add(**_batch)
                        _import_config["batch_imported"] += 1
                        _import_config["samples_imported"] = _import_config["batch_imported"] * \
                            import_batch_size
                        _persist_import_state(
                            _import_state_file, _import_config)
                        console.print(
                            f"Imported {_import_config['samples_imported']} samples", style="bold green")
                        _batch = {
                            "documents": [],
                            "embeddings": [],
                            "metadatas": [],
                            "ids": [],
                        }
            else:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    for i in track(range(0, limit, import_batch_size), description="Importing Dataset"):
                        if _is_resume and i < _import_config["samples_imported"]:
                            continue
                        ds_batch = _dataset[i:min(
                            i + import_batch_size, limit)]
                        if len(ds_batch[document_feature]) == 0:
                            break
                        _batch["documents"].extend(ds_batch[document_feature])
                        if embedding_feature != "None":
                            _batch["embeddings"].extend(
                                ds_batch[embedding_feature])
                        else:
                            del _batch["embeddings"]
                        _batch["ids"].extend(
                            [str(uuid.uuid4()) for _ in range(len(ds_batch[document_feature]))])
                        if len(metadata_features) > 0:
                            _batch["metadatas"].extend([dict(zip(metadata_features, values)) for values in zip(
                                *[ds_batch[feature] for feature in metadata_features])])
                        else:
                            _batch["metadatas"].append(None)
                        # col.add(**_batch)
                        executor.submit(add_to_col, _batch.copy())
                        _import_config["batch_imported"] += 1
                        _import_config["samples_imported"] = min(
                            i + import_batch_size, limit)
                        _persist_import_state(
                            _import_state_file, _import_config)
                        _batch = {
                            "documents": [],
                            "embeddings": [],
                            "metadatas": [],
                            "ids": [],
                        }

            if len(_batch["ids"]) > 0:
                col.add(**_batch)
                _import_config["batch_imported"] += 1
                _import_config["samples_imported"] = _import_config["batch_imported"] * \
                    import_batch_size
                _persist_import_state(_import_state_file, _import_config)
            console.print(
                f"Imported {_import_config['samples_imported']} samples", style="bold green")


@ds_app.command("export")
@ds_app.command("e")
@telemetry_function_decorator
def export_dataset(
        alias: Annotated[Optional[str], typer.Option(
            help="The alias of the environment to use.")] = None,
        type: Annotated[DatasourceType, typer.Option(
            "-t", help="The alias of the environment to use.")] = DatasourceType.HF,
        dataset_path: Annotated[Optional[str], typer.Option(
            "--path", "-d", help="The local path")] = None,
        dataset_file: Annotated[Optional[str], typer.Option(
            "--file", "-f", help="The dataset output compacted file.")] = None,
        collection: Annotated[Optional[str], typer.Option(
            "--collection", "-c", help="Source collection to use.")] = None,
        safe_point_buffer: Annotated[Optional[int], typer.Option(
            help="The maximum results to buffer before persisting to disk.")] = 1000,
        export_batch_size: Annotated[Optional[int], typer.Option(
            help="The size of a batch to export.")] = 500,
        limit: Annotated[Optional[int], typer.Option(
            "--limit", "-l", help="The limit of records to export.")] = -1,
        resume_export_file: Annotated[Optional[str], typer.Option(
            "-c", help="The file to resume export from.")] = None,
        dataset_remote_path: Annotated[Optional[str], typer.Option(
            "-r", help="The remote path to export the dataset to.")] = None,
        dataset_upload: Annotated[Optional[bool], typer.Option(
            "--upload", "-u", help="Upload the dataset to the remote path.")] = False,
):
    if client:
        _is_resume = False
        if resume_export_file:
            _export_config, _export_state_file = _load_export_state(
                resume_export_file)
            if _export_config['finished']:
                confirm = console_interface.confirm(
                    f"Export already completed. Do you want to overwrite it?")
                if not confirm:
                    return
                _export_config['safe_point_offset'] = 0
            _export_id = _export_config["export_id"]
            collection = _export_config["collection"]
            limit = _export_config["limit"]
            export_batch_size = _export_config["export_batch_size"]
            safe_point_buffer = _export_config["safe_point_buffer"]
            dataset_remote_path = _export_config["dataset_remote_path"]
            dataset_upload = _export_config["dataset_upload"]
            dataset_path = _export_config["dataset_path"]
            _is_resume = True
        else:
            _export_id = str(uuid.uuid4())
        col = None
        if not collection:
            collections = client.list_collections()
            collection = console_interface.filter(
                [collection.name for collection in collections], label="Select a collection to export from:")
            col = client.get_collection(collection)
        else:
            col = client.get_collection(collection)
        if not dataset_path:
            dataset_path = console_interface.directory(
                start_dir=Path.home(), label="Select a directory to export the dataset to:")
            dataset_name = console_interface.input(
                placeholder="dataset_name", label="Dataset name:")
            dataset_path = Path(dataset_path).joinpath(dataset_name)

        if Path(dataset_path).exists():
            confirm = console_interface.confirm(
                f"Dataset path {dataset_path} already exists. Do you want to overwrite it?")
            if not confirm:
                return
        else:
            Path(dataset_path).mkdir(parents=True, exist_ok=True)
        col_count = col.count()
        total_results_to_fetch = min(
            limit, col_count) if limit != -1 else col_count
        table = Table(show_header=True,
                      header_style="bold magenta", title="Export Details")
        table.add_column("Collection")
        table.add_column("Export Samples")
        table.add_column("Total Documents in Collection")
        table.add_column("Batch Size")
        table.add_column("Safe Point Buffer")
        table.add_row(collection, str(total_results_to_fetch) if limit != col_count else "all", str(
            col_count), str(export_batch_size), str(safe_point_buffer))
        console.print(table)
        if not _is_resume:
            _export_config, _export_state_file = _create_export_state(_export_id,
                                                                      alias=alias,
                                                                      dataset_path=dataset_path,
                                                                      collection=collection,
                                                                      limit=limit,
                                                                      safe_point_buffer=safe_point_buffer,
                                                                      export_batch_size=export_batch_size,
                                                                      dataset_remote_path=dataset_remote_path,
                                                                      dataset_upload=dataset_upload)
        console.print(
            f"Export state file: {_export_state_file}", style="bold green")
        console.print(f"Exporting to {dataset_path}", style="bold green")

        def read_large_data_in_chunks(offset=0, limit=100):
            result = col.get(
                limit=limit,
                offset=offset,
                include=["embeddings", "documents", "metadatas"])
            return result
        if type == DatasourceType.HF:
            if dataset_upload and not dataset_remote_path:
                dataset_remote_path = console_interface.input(
                    placeholder="<org or user id>/<dataset_id>", label="Remote path to export the dataset to:")
                _persist_export_state(_export_state_file, {
                                      **_export_config, "dataset_remote_path": dataset_remote_path})
            console.print(
                f"Uploading to https://huggingface.co/datasets/{dataset_remote_path}", style="bold green")
            if not console_interface.confirm("Do you want to continue?"):
                return

            data = {
                "id": [],
                "embedding": [],
                "document": [],
                # metadata is added as `metadata.{key}`
            }
            temp_datasets = []
            temp_dateset_paths = []
            metadata_feature = None
            _start = 0
            if _is_resume:
                _start = _export_config["safe_point_offset"]
                console.print(
                    f"Resuming export from offset {_start}", style="bold green")
                temp_dateset_paths = _export_config["safe_point_paths"]
                temp_datasets = [datasets.load_from_disk(
                    path) for path in temp_dateset_paths]
            for offset in track(range(_start, total_results_to_fetch, export_batch_size), description="Exporting Dataset"):
                result = read_large_data_in_chunks(offset=offset, limit=min(
                    total_results_to_fetch-offset, export_batch_size))
                for i, id_ in enumerate(result["ids"]):
                    data["id"].append(str(id_))
                    data["embedding"].append(
                        result["embeddings"][i] if result["embeddings"] else None)
                    data["document"].append(
                        result["documents"][i] if result["documents"] else None)
                    # TODO we need to check if this works if items where metadata is None for a doc
                    if result["metadatas"][i]:
                        for key in result["metadatas"][i].keys():
                            if f"metadata.{key}" not in data:
                                data[f"metadata.{key}"] = []
                            data[f"metadata.{key}"].append(
                                result["metadatas"][i][key])
                        if not metadata_feature:
                            metadata_feature = {}
                        for key in result["metadatas"][i].keys():
                            if key not in metadata_feature:
                                metadata_feature[f"metadata.{key}"] = _infer_hf_type(
                                    result["metadatas"][i][key])
                if len(data["id"]) >= safe_point_buffer:
                    features = datasets.Features({
                        "id": datasets.Value("string"),
                        # TODO this should be int or float
                        "embedding": datasets.features.Sequence(feature=datasets.Value(dtype='float32')),
                        "document": datasets.Value("string"),
                        **metadata_feature
                    })
                    end_offset = offset + \
                        min(total_results_to_fetch-offset, export_batch_size)
                    dataset = Dataset.from_dict(data, features=features, info=datasets.DatasetInfo(
                        description=f"Chroma Dataset for collection {col.name}", features=features))
                    dataset.save_to_disk(str(dataset_path) + f"-{end_offset}")
                    temp_dateset_paths.append(
                        str(dataset_path) + f"-{end_offset}")
                    temp_datasets.append(dataset)
                    _persist_export_state(_export_state_file, {
                                          **_export_config, "safe_point_offset": end_offset, "safe_point_paths": temp_dateset_paths})
                    data = {
                        "id": [],
                        "embedding": [],
                        "document": [],
                        # metadata is added as `metadata.{key}`
                    }

            if len(data["id"]) > 0:
                features = datasets.Features({
                    "id": datasets.Value("string"),
                    # TODO this should be int or float
                    "embedding": datasets.features.Sequence(
                        feature=datasets.Value(dtype='float32')
                        ),
                    "document": datasets.Value("string"),
                    **metadata_feature
                })
                dataset = Dataset.from_dict(data, features=features)
                dataset.save_to_disk(str(dataset_path) + f"-{offset}")
                temp_datasets.append(dataset)

            if len(temp_datasets) >= 1:
                dataset = concatenate_datasets(temp_datasets)
                dataset.save_to_disk(dataset_path)
                for safe_path in temp_dateset_paths:
                    shutil.rmtree(safe_path)
                if dataset_file:
                    dataset.save(dataset_file)
                _persist_export_state(_export_state_file, {
                                      **_export_config, "safe_point_paths": [], "finished": True})
                if dataset_upload:
                    dataset.push_to_hub(dataset_remote_path)
                    custom_metadata = {
                        "license": "mit",
                        "language": "en",
                        "pretty_name": f"Chroma export of collection {col.name}",
                        "size_categories": ["n<1K"],
                        "x-chroma": {
                            "description": f"Chroma Dataset for collection {col.name}",
                            "collection": col.name,
                            "metadata": col.metadata,
                        }}
                    # TODO move this in a separate function
                    card = DatasetCard.load(
                        repo_id_or_path=dataset_remote_path, 
                        repo_type="dataset")
                    data_info = card.data
                    data_dict = {**data_info.to_dict(), **custom_metadata}
                    card.content = f"---\n{str(data_dict)}\n---\n{card.text}"
                    HfApi(endpoint=datasets.config.HF_ENDPOINT).upload_file(
                        path_or_fileobj=str(card).encode(),
                        path_in_repo="README.md",
                        repo_id=dataset_remote_path,
                        repo_type="dataset",
                    )

def eval():
    """evaluate a dataset"""
    #TODO: implement using ragas
    pass

def create():
    """create a new dataset"""
    pass
