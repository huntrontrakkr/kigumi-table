# Start a headless Chrome on the Windows host for tests on the real GPU (from WSL2):
# its own throwaway profile (never your everyday browser) and DevTools on
# 127.0.0.1:<Port>.  cdp.ps1 then drives it.
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File start.ps1 [-Port 9223]
param([int]$Port = 9223)
$exe = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
$profileDir = Join-Path $env:TEMP 'kigumi-cdp-profile'
$p = Start-Process -PassThru -FilePath $exe -ArgumentList @(
  '--headless=new', "--remote-debugging-port=$Port", '--remote-allow-origins=*',
  '--no-first-run', '--no-default-browser-check', '--disable-extensions',
  "--user-data-dir=$profileDir", 'about:blank')
Write-Output ("chrome pid " + $p.Id + ", DevTools on 127.0.0.1:" + $Port)
