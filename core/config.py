from typing import Literal, Optional

from pydantic import PostgresDsn

from pydantic.base_settings BaseSettings


class DBSettings(BaseSettings):
    """Database Settings

    Args:
        BaseSettings (_type_): inherits Core settings.
    """

    DATABASE_URL: PostgresDsn


db = DBSettings()