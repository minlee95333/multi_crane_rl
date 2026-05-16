const localtunnel = require('localtunnel');
(async () => {
  try {
    const tunnel = await localtunnel({ port: 8000 });
    console.log('PUBLIC_URL=' + tunnel.url);
    tunnel.on('close', () => console.log('TUNNEL_CLOSED'));
    setInterval(() => console.log('heartbeat ' + new Date().toISOString()), 30000);
  } catch (err) {
    console.error(err);
    process.exit(1);
  }
})();
