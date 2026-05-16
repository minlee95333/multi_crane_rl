import subprocess, time, os, signal, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
# kill old pids if present
for pidfile in ['/tmp/crane_http.pid','/tmp/crane_lt.pid']:
    try:
        pid=int(Path(pidfile).read_text().strip())
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
server=subprocess.Popen(['python3','-m','http.server','8000','--bind','0.0.0.0'], cwd=ROOT, stdout=open('/tmp/crane_http.log','w'), stderr=subprocess.STDOUT, start_new_session=True)
Path('/tmp/crane_http.pid').write_text(str(server.pid))
time.sleep(1)
lt=subprocess.Popen(['npx','--yes','localtunnel','--port','8000'], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
Path('/tmp/crane_lt.pid').write_text(str(lt.pid))
url=None
lines=[]
start=time.time()
while time.time()-start<20:
    line=lt.stdout.readline()
    if line:
        lines.append(line)
        print(line, end='')
        if 'your url is:' in line:
            url=line.split('your url is:',1)[1].strip()
            Path('/tmp/crane_public_url.txt').write_text(url)
            print('PUBLIC_URL='+url)
            break
    elif lt.poll() is not None:
        break
    else:
        time.sleep(0.1)
if not url:
    print('NO_URL')
    sys.exit(1)
print(f'PIDS server={server.pid} tunnel={lt.pid}')
