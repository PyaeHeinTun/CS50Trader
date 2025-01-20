import json


def read_config() -> dict:
    with open('./botcore/config.json') as f:
        config_data = json.load(f)

    return config_data
