"""Serve the dashboard via http.server and expose it through an SSH reverse
tunnel to localhost.run.

Cross-platform: PID and log files go to tempfile.gettempdir(); the
detached-session flag is only applied on POSIX (Windows uses
CREATE_NEW_PROCESS_GROUP).
"""
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TMP = Path(tempfile.gettempdir())
HTTP_PID = TMP / 'crane_http.pid'
SSH_PID = TMP / 'crane_ssh.pid'
HTTP_LOG = TMP / 'crane_http.log'
PUBLIC_URL = TMP / 'crane_public_url.txt'

IS_POSIX = os.name == 'posix'
PYTHON = 'python3' if IS_POSIX else sys.executable
POPEN_KWARGS = {'start_new_session': True} if IS_POSIX else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}

for pidfile in (HTTP_PID, SSH_PID):
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

cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ServerAliveInterval=30',
       '-R', '80:localhost:8000', 'nokey@localhost.run']
ssh = subprocess.Popen(
    cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, bufsize=1, **POPEN_KWARGS,
)
SSH_PID.write_text(str(ssh.pid))

url = None
buf = []
start = time.time()
while time.time() - start < 25:
    line = ssh.stdout.readline()
    if line:
        buf.append(line)
        print(line, end='')
        m = re.search(r'https://[^\s]+(?:lhr\.life|localhost\.run)', line)
        if m and 'lhr.life' in m.group(0):
            url = m.group(0).rstrip('.,]')
            PUBLIC_URL.write_text(url)
            print('PUBLIC_URL=' + url)
            break
    elif ssh.poll() is not None:
        break
    else:
        time.sleep(0.1)

if not url:
    print('NO_URL')
    print(''.join(buf[-20:]))
    sys.exit(1)
print(f'PIDS server={server.pid} ssh={ssh.pid}')
