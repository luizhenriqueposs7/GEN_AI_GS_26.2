param([switch]$Avaliar)

$ErrorActionPreference = 'Stop'
$pastaRuntime = Join-Path $PSScriptRoot '.runtime'
New-Item -ItemType Directory -Path $pastaRuntime -Force | Out-Null
$nomeModelo = 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
$nomeMotor = 'llama-b12027-bin-win-cpu-x64.zip'
$modelo = Join-Path $pastaRuntime $nomeModelo
$zipMotor = Join-Path $pastaRuntime $nomeMotor
$servidor = Join-Path $pastaRuntime 'llama/llama-server.exe'

function Obter-Arquivo($destino, $url, $sha) {
    if ((Test-Path -LiteralPath $destino) -and
        (Get-FileHash -LiteralPath $destino -Algorithm SHA256).Hash -eq $sha) {
        return
    }
    $temporario = "$destino.partial"
    Write-Host "Baixando $([System.IO.Path]::GetFileName($destino))"
    curl.exe --location --fail --show-error --retry 2 --output $temporario $url
    if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath $temporario -Algorithm SHA256).Hash -ne $sha) {
        throw "Download incompleto ou arquivo divergente: $destino"
    }
    Move-Item -LiteralPath $temporario -Destination $destino -Force
}

Obter-Arquivo $modelo "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/$nomeModelo" '3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597'
Obter-Arquivo $zipMotor "https://github.com/ggml-org/llama.cpp/releases/download/b12027/$nomeMotor" '041f4f252ae8fd36c52be2561a7ad78d3f9775efb99191a81a9cd0e95e4d5dc6'
if (-not (Test-Path -LiteralPath $servidor)) {
    Expand-Archive -LiteralPath $zipMotor -DestinationPath (Join-Path $pastaRuntime 'llama') -Force
}

$chaveAnterior = $env:LLAMA_API_KEY
$env:LLAMA_API_KEY = [guid]::NewGuid().ToString('N')
$processo = $null
Push-Location -LiteralPath $PSScriptRoot
try {
    $argsServidor = @(
        '-m', ".runtime/$nomeModelo", '--host', '127.0.0.1', '--port', '8080',
        '-c', '2048', '-np', '1', '-t', '6', '-ngl', '0',
        '--no-ui', '--cors-origins', 'http://127.0.0.1:8765', '--no-cors-credentials', '--log-disable'
    )
    $processo = Start-Process -FilePath $servidor -ArgumentList $argsServidor -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru
    $pronto = $false
    for ($i = 0; $i -lt 40 -and -not $processo.HasExited; $i++) {
        try {
            $saude = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/health' -TimeoutSec 2
            if ($saude.status -eq 'ok') { $pronto = $true; break }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $pronto -or $processo.HasExited) {
        throw 'O servidor do modelo não iniciou em 127.0.0.1:8080.'
    }
    if ($Avaliar) {
        python -B -m unittest discover
        if ($LASTEXITCODE -ne 0) { throw 'Falha nos testes automatizados.' }
        python -B avaliar.py
    } else {
        Write-Host 'Abra http://127.0.0.1:8765 no navegador. Encerre com Ctrl+C.'
        python -B app.py
    }
    if ($LASTEXITCODE -ne 0) { throw 'A execução retornou um erro.' }
} finally {
    if ($null -ne $processo -and -not $processo.HasExited) {
        $processo.Kill()
    }
    $env:LLAMA_API_KEY = $chaveAnterior
    Pop-Location
}
