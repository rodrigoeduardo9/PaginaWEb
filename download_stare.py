import os
from urllib import request
import zipfile

OUT='external_datasets'
os.makedirs(OUT, exist_ok=True)
url='http://cecas.clemson.edu/~ahoover/stare/images/all-images.zip'
out_path=os.path.join(OUT,'stare_all_images.zip')
print('Downloading',url)
request.urlretrieve(url,out_path)
print('Downloaded to',out_path)
print('Extracting...')
zipfile.ZipFile(out_path).extractall(os.path.join(OUT,'stare'))
print('Extracted to',os.path.join(OUT,'stare'))
