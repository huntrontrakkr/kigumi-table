# Drive an already running headless Chrome (isolated profile, DevTools on 127.0.0.1:9223):
# open a page, evaluate an async JS expression, screenshot.  Used from WSL to test
# the kigumi browser build on the real GPU.
param([string]$Url, [string]$Expr, [string]$Shot = "C:\Temp\kigumi-shot.png", [int]$TimeoutS = 600)
$ErrorActionPreference = "Stop"
$t = Invoke-RestMethod -Method Put -Uri ("http://127.0.0.1:9223/json/new?" + $Url)
$ws = New-Object System.Net.WebSockets.ClientWebSocket
$ws.Options.KeepAliveInterval = [TimeSpan]::FromSeconds(10)
$ws.ConnectAsync([Uri]$t.webSocketDebuggerUrl, [Threading.CancellationToken]::None).Wait()
$script:id = 0
function Send($method, $params) {
  $script:id++
  $msg = @{ id = $script:id; method = $method; params = $params } | ConvertTo-Json -Depth 20 -Compress
  $bytes = [Text.Encoding]::UTF8.GetBytes($msg)
  $ws.SendAsync([ArraySegment[byte]]$bytes, [Net.WebSockets.WebSocketMessageType]::Text, $true, [Threading.CancellationToken]::None).Wait()
  $want = $script:id
  $buf = New-Object byte[] 1048576
  while ($true) {
    $sb = New-Object Text.StringBuilder
    do {
      $cts = New-Object Threading.CancellationTokenSource ([TimeSpan]::FromSeconds($TimeoutS))
      $r = $ws.ReceiveAsync([ArraySegment[byte]]$buf, $cts.Token).Result
      [void]$sb.Append([Text.Encoding]::UTF8.GetString($buf, 0, $r.Count))
    } while (-not $r.EndOfMessage)
    $o = $sb.ToString() | ConvertFrom-Json
    if ($o.id -eq $want) { return $o }
  }
}
[void](Send "Page.enable" @{})
[void](Send "Runtime.enable" @{})
Start-Sleep -Seconds 2
$res = Send "Runtime.evaluate" @{ expression = $Expr; awaitPromise = $true; returnByValue = $true; timeout = ($TimeoutS * 1000) }
if ($res.result.exceptionDetails) { Write-Output ("EXCEPTION: " + ($res.result.exceptionDetails | ConvertTo-Json -Depth 6 -Compress)) }
else { Write-Output ("RESULT: " + ($res.result.result.value | ConvertTo-Json -Depth 8 -Compress)) }
$shotRes = Send "Page.captureScreenshot" @{ format = "png" }
[IO.File]::WriteAllBytes($Shot, [Convert]::FromBase64String($shotRes.result.data))
Write-Output ("SHOT: " + $Shot)
[void](Invoke-RestMethod -Uri ("http://127.0.0.1:9223/json/close/" + $t.id))
