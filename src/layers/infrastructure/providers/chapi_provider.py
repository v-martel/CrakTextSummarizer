from src.shared.config.configuration import configuration
from src.layers.infrastructure.providers.temporary.queries import queryMfcTagsAndTitles

import requests
import json


class ChapiProvider:
    def __init__(self):
        self.config = configuration.layers.infrastructure.providers.chapi_provider

    def execute_query(self) -> dict:
        response = requests.post(
            url=self.config.chapi_url,
            json={'query': queryMfcTagsAndTitles(self.config.chapi_token)}
        )

        return json.loads(response.content.decode('utf-8'))['data'] if response.status_code == 200 else None
