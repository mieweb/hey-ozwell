// Tiny static server for the isolated speaker-verification WASM test.
// Sets the COOP/COEP headers required for SharedArrayBuffer (the WASM build uses
// pthreads). Kept separate from the main demo server so it can't break the
// existing page's cross-origin loads. Run: node serve-coi.js  ->  :3010/test.html
const http = require('http'), fs = require('fs'), path = require('path');
const root = __dirname, port = process.env.PORT || 3010;
const mime = {
  '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm',
  '.data': 'application/octet-stream', '.onnx': 'application/octet-stream',
  '.wav': 'audio/wav', '.json': 'application/json', '.css': 'text/css',
};
http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p === '/') p = '/test.html';
  const fp = path.join(root, p);
  // cross-origin isolation (required for SharedArrayBuffer / wasm threads)
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  fs.readFile(fp, (e, data) => {
    if (e) { res.statusCode = 404; res.end('not found: ' + p); return; }
    res.setHeader('Content-Type', mime[path.extname(fp)] || 'application/octet-stream');
    res.end(data);
  });
}).listen(port, () => console.log(`COI test server: http://localhost:${port}/test.html`));
