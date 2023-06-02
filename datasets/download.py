from six.moves import urllib
import os
import sys

def download(url):
	filename = url.split('/')[-1]
	savepath=filename
	urllib.request.urlretrieve(url, savepath)

with open('url.txt') as f:
	files = f.readlines()
	files = [item.split('\n')[0] for item in files]

for url in files:
	download(url)
