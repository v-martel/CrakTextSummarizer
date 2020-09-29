from src.shared.config.configuration import configuration
from src.layers.infrastructure.providers.temporary.queries import queryMfcTagsAndTitles, queryMfcTotal

import requests
import json


class ChapiProvider:
    def __init__(self):
        self.config = configuration.layers.infrastructure.providers.chapi_provider

    def get_total_vids_number(self):
        response = requests.post(
            url=self.config.chapi_url,
            json={'query': queryMfcTotal(self.config.chapi_token)}
        )

        return json.loads(response.content.decode('utf-8'))['data'] if response.status_code == 200 else None

    def get_vids_at_page(self, page: str) -> dict:
        response = requests.post(
            url=self.config.chapi_url,
            json={'query': queryMfcTagsAndTitles(self.config.chapi_token, page)}
        )

        return json.loads(response.content.decode('utf-8'))['data'] if response.status_code == 200 else None
