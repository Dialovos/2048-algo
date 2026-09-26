#!/usr/bin/env node
// Local training dashboard: trains networks on this computer's cores, shows
// progress, and hands out weights for the panel's "Load weights".
//
// Usage: node dashboard.js [--port N] [--dir FOLDER] [--no-open]
//   --port N      port on 127.0.0.1 (default 2048)
//   --dir FOLDER  where networks live (default weights/ next to this file)
//   --no-open     don't open the browser
// Ctrl+C saves any training in progress and exits.

const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFile } = require('child_process');
const { startTraining, listNetworks, readHistory, DEFAULTS } = require('./train.js');

const args = process.argv.slice(2);
const argValue = (flag, fallback) => (args.includes(flag) ? args[args.indexOf(flag) + 1] : fallback);
const PORT = Number(argValue('--port', 2048));
const DIR = path.resolve(argValue('--dir', path.join(__dirname, 'weights')));
const PAGE = path.join(__dirname, 'dashboard.html');
const USERSCRIPT = path.join(__dirname, '2048-algo.user.js');
const NAME = /^[a-z0-9][a-z0-9-]{0,62}$/; // network names map straight to files in DIR

let trainer = null;
let lastError = '';
const lines = [];

function note(line) {
  console.log(line);
  lines.push(`${new Date().toLocaleTimeString()}  ${line}`);
  if (lines.length > 200) lines.shift();
}

function state() {
  return {
    cores: os.cpus().length,
    memoryGB: os.totalmem() / 2 ** 30,
    dir: DIR,
    training: trainer && !trainer.stopped ? {
      name: path.basename(trainer.files.bin, '.bin'),
      tc: trainer.useTc,
      alpha: trainer.alpha,
      threads: trainer.threads,
      restart: trainer.restart,
      startedAt: trainer.startedAt,
      summary: trainer.summary(),
    } : null,
    networks: listNetworks(DIR),
    lines: lines.slice(-40),
    error: lastError,
  };
}

function start(body) {
  if (trainer && !trainer.stopped) throw new Error('Already training; stop first.');
  const name = String(body.name || '');
  if (!NAME.test(name)) throw new Error('Use lowercase letters, digits and dashes for the name.');
  const threads = Math.round(Number(body.threads));
  if (!(threads >= 1 && threads <= os.cpus().length)) throw new Error(`Threads must be 1 to ${os.cpus().length}.`);
  const restart = Number(body.restart);
  if (!(restart >= 0 && restart <= 0.9)) throw new Error('Restart share must be 0 to 0.9.');
  const alpha = body.alpha === '' || body.alpha == null ? NaN : Number(body.alpha);
  if (!Number.isNaN(alpha) && !(alpha > 0 && alpha <= 1)) throw new Error('Step size must be above 0 and at most 1, or blank for automatic.');
  lastError = '';
  // tc and optimistic only matter for a new network; existing ones keep theirs.
  trainer = startTraining({
    weights: path.join(DIR, `${name}.bin`), threads, restart, alpha, reportSeconds: 30,
    noTc: body.tc === false, optimistic: body.optimistic === false ? 0 : DEFAULTS.optimistic,
  }, note);
}

function send(res, status, body, type = 'text/plain; charset=utf-8') {
  res.writeHead(status, { 'Content-Type': type, 'Cache-Control': 'no-store' });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => {
      data += chunk;
      if (data.length > 10000) req.destroy();
    });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  // Only this computer's own pages: the Host check stops other sites reaching
  // this server through DNS tricks, and requiring JSON posts makes browsers
  // ask first (CORS preflight), which this server never approves.
  if (req.headers.host !== `127.0.0.1:${PORT}` && req.headers.host !== `localhost:${PORT}`) return send(res, 403, 'Forbidden');
  const url = new URL(req.url, `http://${req.headers.host}`);

  try {
    if (req.method === 'GET') {
      if (url.pathname === '/') return send(res, 200, fs.readFileSync(PAGE), 'text/html; charset=utf-8');
      if (url.pathname === '/api/state') return send(res, 200, JSON.stringify(state()), 'application/json');
      if (url.pathname === '/api/history') {
        const name = url.searchParams.get('name') || '';
        if (!NAME.test(name)) return send(res, 400, 'Bad name');
        return send(res, 200, JSON.stringify(readHistory(path.join(DIR, `${name}.bin`))), 'application/json');
      }
      if (url.pathname === '/2048-algo.user.js') return send(res, 200, fs.readFileSync(USERSCRIPT), 'text/javascript; charset=utf-8');
      const download = /^\/weights\/([a-z0-9][a-z0-9-]{0,62})\.bin$/.exec(url.pathname);
      if (download) {
        const file = path.join(DIR, `${download[1]}.bin`);
        if (!fs.existsSync(file)) return send(res, 404, 'No such network');
        res.writeHead(200, {
          'Content-Type': 'application/octet-stream',
          'Content-Length': fs.statSync(file).size,
          'Content-Disposition': `attachment; filename="${download[1]}.bin"`,
        });
        return fs.createReadStream(file).pipe(res);
      }
      return send(res, 404, 'Not found');
    }

    if (req.method === 'POST') {
      if (!/^application\/json\b/.test(req.headers['content-type'] || '')) return send(res, 415, 'JSON only');
      const body = JSON.parse((await readBody(req)) || '{}');
      if (url.pathname === '/api/start') start(body);
      else if (url.pathname === '/api/stop') trainer && trainer.stop();
      else if (url.pathname === '/api/save') {
        if (!trainer || trainer.stopped) throw new Error('Not training.');
        trainer.save();
      } else return send(res, 404, 'Not found');
      return send(res, 200, JSON.stringify(state()), 'application/json');
    }
    return send(res, 405, 'Method not allowed');
  } catch (err) {
    lastError = err.message;
    note(`Error: ${err.message}`);
    return send(res, 400, JSON.stringify({ error: err.message }), 'application/json');
  }
});

server.on('error', (err) => {
  // Suggest a port browsers accept (2049, for one, is on their blocked list).
  console.error(err.code === 'EADDRINUSE' ? `Port ${PORT} is busy; try --port 8048.` : err.message);
  process.exit(1);
});

server.listen(PORT, '127.0.0.1', () => {
  const address = `http://127.0.0.1:${PORT}`;
  console.log(`Training dashboard: ${address}  (Ctrl+C to quit; training in progress is saved first)`);
  if (args.includes('--no-open')) return;
  const [cmd, cmdArgs] = process.platform === 'win32' ? ['cmd', ['/c', 'start', '', address]]
    : process.platform === 'darwin' ? ['open', [address]] : ['xdg-open', [address]];
  execFile(cmd, cmdArgs, () => {});
});

process.on('SIGINT', () => {
  if (trainer) trainer.stop();
  process.exit(0);
});
