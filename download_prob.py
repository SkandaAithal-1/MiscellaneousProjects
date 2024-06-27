#!/usr/bin/env python3

"""Download and setup problems from Competitive Companion
Usage:
  download_prob.py [<name>... | -n <number>]  
  download_prob.py --echo
  
Options:
  -h --help     Show this screen.
  --echo        Just echo the thing
  -n COUNT, --number COUNT   Number of problems. [default: 0]
"""

from docopt import docopt

import json
import sys
import http.server
import subprocess
import re
from pathlib import Path


def listenOnce(*, timeout=None):
    json_data = None

    class CompetitiveCompanionHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal json_data
            json_data = json.load(self.rfile)
            print("Here")
            print(json_data)

    with http.server.HTTPServer(('127.0.0.1', 10045), CompetitiveCompanionHandler) as server:
        server.timeout = timeout
        server.handle_request()

    print(json_data)
    if json_data is not None:
        print(f"Data received")
    else:
        print(f"Data not received")

    return json_data

NAME_PATTERN = re.compile(r'^(?:Problem )?([A-Z][0-9]*)\b')

def listenMany(*, numItems=None, timeout=None):
    if numItems is not None:
        res = []

        for _ in range(numItems):
            cur = listenOnce()
            res.append(cur)
            
        return res
    

def getProbName(data):
    return data['name']

def saveSamples(data, probDir):
    with open(probDir / 'problem.json', 'w') as f:
        json.dump(data, f)

    for i, t in enumerate(data['tests'], start=1):
        with open(probDir / f'in{i}', 'w') as f:
            f.write(t['input'])
        with open(probDir / f'out{i}', 'w') as f:
            f.write(t['output'])

def makeProb(data, name=None):
    if name is None:
        name = getProbName(data)
    probDir = Path('.') / name

    if probDir.exists() and probDir.is_dir():
        print(f"Already created problem {name}...")
    else:
        print(f"Creating problem {name}...")

        MAKE_PROB = Path(sys.path[0]) / 'make_prob.sh'

        try:
            subprocess.check_call([MAKE_PROB, name], stdout=sys.stdout, stderr=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Got error {e}")
            return 

    print("Saving samples...")
    saveSamples(data, probDir)

    print()

    
def main():
    
    args = docopt(__doc__)

    if args['--echo']:
        while True:
            datas = listenOnce()
            print(datas)

    if names := args['<name>']:
        datas = listenMany(numItems=len(names))
        for data, name in zip(datas, names):
            makeProb(data, name)
    
    elif cnt := args['--number']:
        cnt = int(cnt)
        datas = listenMany(numItems=cnt)

        for data in datas:
            makeProb(data)

if __name__=="__main__":
    main()
