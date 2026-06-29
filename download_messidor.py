"""Try to download Messidor by scraping the official page for .zip links and fetching them.
Saves to `external_datasets/`.
"""
import os
import re
import sys
from urllib import request, parse


ROOT_PAGE = 'http://www.adcis.net/en/Download-Third-Party/Messidor.html'
OUT_DIR = os.path.join(os.path.dirname(__file__), 'external_datasets')


def fetch(url):
    print('Fetching', url)
    req = request.Request(url, headers={'User-Agent': 'python-urllib/3'})
    with request.urlopen(req, timeout=30) as r:
        return r.read()


def find_zip_links(html, base_url):
    text = html.decode('utf-8', errors='ignore')
    hrefs = re.findall(r'href=["\']([^"\']+\.zip)["\']', text, flags=re.IGNORECASE)
    links = []
    for h in hrefs:
        links.append(parse.urljoin(base_url, h))
    return links


def download_file(url, out_path):
    print('Downloading', url)
    req = request.Request(url, headers={'User-Agent': 'python-urllib/3'})
    with request.urlopen(req, timeout=60) as r, open(out_path, 'wb') as f:
        data = r.read()
        f.write(data)
    print('Saved to', out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        html = fetch(ROOT_PAGE)
    except Exception as e:
        print('Failed to fetch page:', e)
        return
    links = find_zip_links(html, ROOT_PAGE)
    if not links:
        print('No .zip links found on the page.')
        return
    for i, link in enumerate(links):
        fname = os.path.basename(parse.urlsplit(link).path)
        out_path = os.path.join(OUT_DIR, fname)
        try:
            download_file(link, out_path)
            print('Downloaded', link)
            return
        except Exception as e:
            print('Failed to download', link, 'error:', e)
    print('All attempts failed.')


if __name__ == '__main__':
    main()
