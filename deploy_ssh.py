import subprocess, time, os, signal, sys, re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for pidfile in ['/tmp/crane_http.pid','/tmp/crane_ssh.pid']:
    try:
        pid=int(Path(pidfile).read_text().strip()); os.kill(pid, signal.SIGTERM)
    except Exception: pass
server=subprocess.Popen(['python3','-m','http.server','8000','--bind','0.0.0.0'], cwd=ROOT, stdout=open('/tmp/crane_http.log','w'), stderr=subprocess.STDOUT, start_new_session=True)
Path('/tmp/crane_http.pid').write_text(str(server.pid))
time.sleep(1)
cmd=['ssh','-o','StrictHostKeyChecking=no','-o','ServerAliveInterval=30','-R','80:localhost:8000','nokey@localhost.run']
ssh=subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
Path('/tmp/crane_ssh.pid').write_text(str(ssh.pid))
url=None; start=time.time(); buf=[]
while time.time()-start<25:
    line=ssh.stdout.readline()
    if line:
        buf.append(line); print(line, end='')
        m=re.search(r'https://[^\s]+(?:lhr\.life|localhost\.run)', line)
        if m and 'lhr.life' in m.group(0):
            url=m.group(0).rstrip('.,]')
            Path('/tmp/crane_public_url.txt').write_text(url)
            print('PUBLIC_URL='+url)
            break
    elif ssh.poll() is not None: break
    else: time.sleep(0.1)
if not url:
    print('NO_URL')
    print(''.join(buf[-20:]))
    sys.exit(1)
print(f'PIDS server={server.pid} ssh={ssh.pid}')
