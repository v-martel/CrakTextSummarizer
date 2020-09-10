from src.shared.config.configuration import configuration
from src.layers.infrastructure.providers.temporary.queries import queryMfcTagsAndTitles

import requests
import json


class ChapiProvider:
    def __init__(self):
        self.config = configuration.layers.infrastructure.providers.chapi_provider

    def execute_query(self):
        response = requests.post(
            url=self.config.chapi_url,
            json={'query': queryMfcTagsAndTitles}
        )

        if response.status_code == 200:
            return json.loads(response.content.decode('utf-8'))
        return None
