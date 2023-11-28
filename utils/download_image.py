# download image
import os
import requests
import http.client
import time


FRAGRANTICA_IMAGE_DIR= "/dbfs/mnt/stg/house_of_fragrance/fragrantica_images"
os.makedirs(FRAGRANTICA_IMAGE_DIR, exist_ok=True)


def get_image_name(product_name, image_url):
    _id = image_url.split(".")[-2]
    product_name = product_name.replace(" ", "_")
    product_name = product_name.replace("/", "_")
    return f"{product_name}__{_id}.jpg"


def _retry_get_request(url, tries: int=-1, delay: float=0, max_delay: float=None, backoff: int=1):
    """
    Retries `requests.get(url)` automatically.
    Copied from `retry` package: https://github.com/invl/retry

    :param url:
    :param tries: the maximum number of attempts. default: -1 (infinite).
    :param delay: initial delay between attempts. default: 0.
    :param max_delay: the maximum value of delay. default: None (no limit).
    :param backoff: multiplier applied to delay between attempts. default: 1 (no backoff).
    :returns: the response from url
    """
    while tries:
        try:
            return requests.get(url, stream=True)
        except (requests.RequestException, http.client.IncompleteRead):
            tries -= 1
            if not tries:
                raise

            time.sleep(delay)
            delay *= backoff

            if max_delay is not None:
                delay = min(delay, max_delay)


def download_image(image_url, product_name, image_dir):
    image_name = get_image_name(product_name, image_url)
    dest_path = os.path.join(image_dir, image_name)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return

    with _retry_get_request(image_url, tries=5, delay=1, max_delay=10, backoff=2) as r:
        if r.status_code != 200:
            return

        r.raw.decode_content = True
        content = r.raw.read()
        if len(content) == 0:
            return

        with open(dest_path, 'wb') as f:
            f.write(content)
    return image_name
