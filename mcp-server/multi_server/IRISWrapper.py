import os
import iris
from contextlib import contextmanager

def get_iris_config():
    """Get database configuration from environment variables."""
    config = {
        "hostname": os.getenv("IRIS_HOSTNAME", "localhost"),
        "port": int(os.getenv("IRIS_PORT", 1980)),
        "namespace": os.getenv("IRIS_NAMESPACE", "MCP"),
        "username": os.getenv("IRIS_USERNAME", "superuser"),
        "password": os.getenv("IRIS_PASSWORD", "SYS")
    }

    #logger.info("Server configuration: iris://" + config["hostname"] + ":" + str(config["port"]) + "/" + config["namespace"])
    if not all([config["username"], config["password"], config["namespace"]]):
        raise ValueError("Missing required database configuration")

    return config
