// Run the kigumi package under Pyodide in Node, with Manifold's WebAssembly
// build behind site/engine/manifold-bridge.js -- the same engine the browser
// worker runs, without a browser.   node run.mjs [preset]
import { loadPyodide } from 'pyodide';
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
const presetName = process.argv[2] || '';

const t0 = performance.now();
const pyodide = await loadPyodide({ packageCacheDir: path.join(here, '.pyodide-cache') });
await pyodide.loadPackage(['numpy', 'scipy', 'shapely', 'contourpy'], { messageCallback: () => {} });
const t1 = performance.now();

const { initManifold } = await import(pathToFileURL(path.join(root, 'site/engine/manifold-bridge.js')).href);
const bridge = await initManifold(pathToFileURL(path.join(here, 'node_modules/manifold-3d/manifold.js')).href);
pyodide.registerJsModule('kigumi_manifold_js', bridge);

// the package sources into the Pyodide file system
pyodide.FS.mkdirTree('/home/pyodide/kigumi');
for (const f of fs.readdirSync(path.join(root, 'kigumi'))) {
  if (f.endsWith('.py')) pyodide.FS.writeFile(`/home/pyodide/kigumi/${f}`, fs.readFileSync(path.join(root, 'kigumi', f)));
}
const t2 = performance.now();

let out;
try {
out = await pyodide.runPythonAsync(`
import json, time, sys
sys.path.insert(0, "/home/pyodide")
from kigumi import webapi
from kigumi.params import preset
name = ${JSON.stringify(presetName)}
params = preset(name).to_dict() if name else {}
t = time.time(); pub, glb = webapi.build(json.dumps({"params": params})); tb = time.time() - t
rep = json.loads(pub)
t = time.time(); c = json.loads(webapi.checks(rep["id"])); tc = time.time() - t
from kigumi.linalg import STATS
json.dumps(dict(build_s=round(tb, 2), checks_s=round(tc, 2), glb=len(glb), parts=rep["report"].get("n_parts"),
                notes=rep["report"].get("notes"),
                checks=[(x["name"], x["status"], x["value"]) for x in c["checks"]], linalg=STATS), default=float)
`);
} catch (e) {
  console.log('ERROR:', String(e.message || e).slice(-3000));
  process.exit(1);
}
const t3 = performance.now();
console.log(`pyodide+packages ${((t1 - t0) / 1000).toFixed(1)}s, manifold+sources ${((t2 - t1) / 1000).toFixed(1)}s, run ${((t3 - t2) / 1000).toFixed(1)}s`);
console.log(out);
