const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const { setTimeout: delay } = require('node:timers/promises');
const vm = require('node:vm');

const ROOT = path.resolve(__dirname, '..');
async function waitFor(check, timeout = 60000) {
  const until = Date.now()+timeout;
  while (Date.now() < until) { const result = await check(); if (result) return result; await delay(100); }
  throw new Error('Timed out waiting for dashboard/training state.');
}

test('dashboard CPU/GPU switch, progress, save, and shutdown', {timeout: 180000}, async t => {
  fs.mkdirSync(path.join(ROOT, '.scratch'), {recursive: true});
  const directory = fs.mkdtempSync(path.join(ROOT, '.scratch', 'dashboard-test-'));
  const reserved = http.createServer(); reserved.listen(0, '127.0.0.1'); await once(reserved, 'listening');
  const port = reserved.address().port; await new Promise(resolve => reserved.close(resolve));
  const child = spawn(process.execPath, ['dashboard.js', '--port', String(port), '--dir', directory, '--no-open'], {cwd: ROOT});
  let output = '';
  child.stdout.on('data', chunk => { output += chunk; });
  child.stderr.on('data', chunk => { output += chunk; });
  const closed = once(child, 'close');
  t.after(async () => {
    if (child.exitCode === null) child.kill('SIGTERM');
    await closed;
    fs.rmSync(directory, {recursive: true, force: true});
  });
  const base = `http://127.0.0.1:${port}`;
  async function api(route, body) {
    const response = await fetch(base+route, body === undefined ? {} : {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
    });
    return {code: response.status, body: await response.json()};
  }
  await waitFor(async () => { try { return (await api('/api/state')).code === 200; } catch (_) { return false; } });
  const html = await (await fetch(base)).text();
  assert.match(html, /id="backend"/);
  assert.match(html, /value="gpu"/);
  new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
  const forbidden = await new Promise((resolve, reject) => {
    http.get(base+'/api/state', {headers: {Host: 'example.invalid'}}, response => {
      response.resume(); resolve(response.statusCode);
    }).on('error', reject);
  });
  assert.equal(forbidden, 403);
  const invalid = await api('/api/start', {name: 'test', backend: 'unknown'});
  assert.equal(invalid.code, 400);
  const settings = {name: 'switch', backend: 'cpu', threads: 1, tc: true, optimistic: true, alpha: '', restart: 0.5};
  let result = await api('/api/start', settings);
  assert.equal(result.code, 200);
  assert.equal(result.body.training.backend, 'cpu');
  assert.equal((await api('/api/start', settings)).code, 400);
  await delay(1500);
  assert.equal((await api('/api/save', {})).code, 200);
  result = await api('/api/stop', {});
  assert.equal(result.code, 200);
  assert.equal(result.body.training, null);
  const readTotals = () => JSON.parse(fs.readFileSync(path.join(directory, 'switch.json')));
  const cpuTotals = readTotals();
  assert.ok(cpuTotals.games > 0);
  assert.equal(fs.statSync(path.join(directory, 'switch.bin')).size, 268435456);
  assert.equal(fs.statSync(path.join(directory, 'switch.tc')).size, 536870912);
  const capability = await waitFor(async () => { const s = (await api('/api/state')).body; return !s.gpu.checking && s.gpu; });
  if (!capability.available) { t.skip(capability.reason); return; }
  result = await api('/api/start', {...settings, backend: 'gpu', gpuGames: 32});
  assert.equal(result.code, 200);
  assert.equal(result.body.training.backend, 'gpu');
  await waitFor(async () => {
    const s = (await api('/api/state')).body;
    if (s.error) throw new Error(s.error+'\n'+output);
    return s.training && s.training.status === 'training' && s.training.summary.games > cpuTotals.games;
  });
  result = await api('/api/save', {});
  assert.equal(result.code, 200);
  assert.equal(result.body.training.tc, true);
  assert.ok(result.body.training.summary.movesPerSecond > 0);
  assert.equal((await api('/api/stop', {})).code, 200);
  const gpuTotals = readTotals(); assert.ok(gpuTotals.games > cpuTotals.games);
  result = await api('/api/start', settings);
  assert.equal(result.code, 200); assert.equal(result.body.training.tc, true);
  await delay(1500);
  assert.equal((await api('/api/stop', {})).code, 200);
  assert.ok(readTotals().games > gpuTotals.games);
  // Closing the dashboard with a live GPU worker must wait for its final save.
  assert.equal((await api('/api/start', {...settings, backend: 'gpu', gpuGames: 32})).code, 200);
  await waitFor(async () => (await api('/api/state')).body.training?.status === 'training');
  child.kill('SIGTERM');
  const [exitCode] = await closed;
  assert.equal(exitCode, 0, output);
  assert.doesNotThrow(readTotals);
  assert.equal(fs.readdirSync(directory).filter(name => name.endsWith('.tmp')).length, 0);
});
