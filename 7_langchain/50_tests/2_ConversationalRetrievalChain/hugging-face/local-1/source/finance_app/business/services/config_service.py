import configparser
from enum import Enum
import os
from utils.files import get_full_path


class Environment(Enum):
    DEV = "dev"
    PROD = "prod"


class AppConfig(object):
    def __init__(self) -> None:
        super().__init__()

        self.config = configparser.ConfigParser()
        self.config.read(get_full_path("config.ini"))

    def environment(self) -> str:
        if "APP_SETTINGS" in self.config:
            if "environment" in self.config["APP_SETTINGS"]:
                return self.config["APP_SETTINGS"]["environment"]
            else:
                raise Exception("THIS SHOULD NOT HAPPENED")
        else:
            raise Exception("THIS SHOULD NOT HAPPENED")

    def twsIP(self) -> str:
        if "IB" in self.config:
            if "tws_ip" in self.config["IB"]:
                return self.config["IB"]["tws_ip"]
            else:
                raise Exception("THIS SHOULD NOT HAPPENED")
        else:
            raise Exception("THIS SHOULD NOT HAPPENED")

    def twsPort(self) -> str:
        env = self.environment()
        if "IB" in self.config:
            if env == Environment.DEV.value:
                if "tws_sim_port" in self.config["IB"]:
                    return self.config["IB"]["tws_sim_port"]
                else:
                    raise Exception("THIS SHOULD NOT HAPPENED")
            elif env == Environment.PROD.value:
                if "tws_real_port" in self.config["IB"]:
                    return self.config["IB"]["tws_real_port"]
                else:
                    raise Exception("THIS SHOULD NOT HAPPENED")
            else:
                raise Exception("THIS SHOULD NOT HAPPENED")
        else:
            raise Exception("THIS SHOULD NOT HAPPENED")
