"""Serve the dashboard via http.server and expose it through localtunnel.

Cross-platform: PID and log files go to tempfile.gettempdir(), and the
detached-session flag is only applied on POSIX (Windows uses
CREATE_NEW_PROCESS_GROUP). Run from any OS that has python3 and npx.
"""
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TMP = Path(tempfile.gettempdir())
HTTP_PID = TMP / 'crane_http.pid'
LT_PID = TMP / 'crane_lt.pid'
HTTP_LOG = TMP / 'crane_http.log'
PUBLIC_URL = TMP / 'crane_public_url.txt'

IS_POSIX = os.name == 'posix'
PYTHON = 'python3' if IS_POSIX else sys.executable
NPX = 'npx' if IS_POSIX else 'npx.cmd'
POPEN_KWARGS = {'start_new_session': True} if IS_POSIX else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}

for pidfile in (HTTP_PID, LT_PID):
    try:
        pid = int(pidfile.read_text().strip())
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass

server = subprocess.Popen(
    [PYTHON, '-m', 'http.server', '8000', '--bind', '0.0.0.0'],
    cwd=ROOT, stdout=HTTP_LOG.open('w'), stderr=subprocess.STDOUT, **POPEN_KWARGS,
)
HTTP_PID.write_text(str(server.pid))
time.sleep(1)

lt = subprocess.Popen(
    [NPX, '--yes', 'localtunnel', '--port', '8000'],
    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **POPEN_KWARGS,
)
LT_PID.write_text(str(lt.pid))

url = None
lines = []
start = time.time()
while time.time() - start < 20:
    line = lt.stdout.readline()
    if line:
        lines.append(line)
        print(line, end='')
        if 'your url is:' in line:
            url = line.split('your url is:', 1)[1].strip()
            PUBLIC_URL.write_text(url)
            print('PUBLIC_URL=' + url)
            break
    elif lt.poll() is not None:
        break
    else:
        time.sleep(0.1)

if not url:
    print('NO_URL')
    sys.exit(1)
print(f'PIDS server={server.pid} tunnel={lt.pid}')
