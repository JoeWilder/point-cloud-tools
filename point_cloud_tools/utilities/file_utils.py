import os
import requests
import zipfile


def download(url: str, path: str):
    """Download file using the given url"""
    get_response = requests.get(url, stream=True)
    filename = url.split("/")[-1].split("?")[0]

    if path is not None:
        filename = os.path.join(path, filename)

    if not os.path.exists(path):
        os.mkdir(path)

    with open(filename, "wb") as f:
        for chunk in get_response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip(zip_file_path, extract_path, clean=True):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    if clean:
        os.remove(zip_file_path)
