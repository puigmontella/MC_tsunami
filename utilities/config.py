# utilities/config.py
import os
import json

def load_config(path: str) -> dict:
    path_abs = os.path.abspath(path)
    with open(path_abs, "r") as f:
        cfg = json.load(f)

    # Store where the config lives
    cfg["_config_path"] = path_abs

    # Define project root as parent of the directory containing the config
    # e.g. .../MC_tsunami/cases/config.json -> root = .../MC_tsunami
    cfg["_root"] = os.path.abspath(os.path.join(os.path.dirname(path_abs), ".."))
    return cfg


def abspath(cfg: dict, rel_or_abs: str) -> str:
    if rel_or_abs is None:
        return None
    if os.path.isabs(rel_or_abs):
        return rel_or_abs

    root = cfg.get("_root")
    if not root:
        # fallback: parent of config dir
        cpath = os.path.abspath(cfg.get("_config_path", "."))
        root = os.path.abspath(os.path.join(os.path.dirname(cpath), ".."))

    return os.path.abspath(os.path.join(root, rel_or_abs))
