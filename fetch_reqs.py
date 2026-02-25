import urllib.request as r
print(r.urlopen('https://raw.githubusercontent.com/m-bain/whisperX/main/requirements.txt').read().decode())
