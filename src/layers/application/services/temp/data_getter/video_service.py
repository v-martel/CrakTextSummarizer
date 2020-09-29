from time import sleep
from src.layers.infrastructure.providers.chapi_provider import ChapiProvider


class VideoService:
    def __init__(self):
        self.chapi_provider = ChapiProvider()

    def get_all_vids(self):
        total_vids = int(self.chapi_provider.get_total_vids_number()['PremiumVideos']['Pager']['total'])
        print(total_vids)
        sleep(10)
        step_size = 10000
        total_steps = int(total_vids/step_size)
        page = "0"
        ans = []

        for _ in range(total_steps):
            data = self.chapi_provider.get_vids_at_page(page)['PremiumVideos']
            print(len(data['Data']), data['Pager']['nextPage'])
            page = data['Pager']['nextPage']

            ans = ans + data['Data']
            sleep(10)

        return ans
