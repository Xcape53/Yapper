param(
    [string]$DistPath = 'release_staging\groq',
    [string]$WorkPath = 'build\groq'
)

$ErrorActionPreference = 'Stop'
$previousPath = $env:PATH
Push-Location $PSScriptRoot
try {
    # Avoid bundling unrelated DLLs from developer tools present on PATH.
    # PyInstaller discovers Python and package-specific DLL directories itself.
    $env:PATH = @(
        (Join-Path $PSScriptRoot '.venv\Scripts'),
        (Join-Path $env:SystemRoot 'System32'),
        $env:SystemRoot
    ) -join [IO.Path]::PathSeparator
    & '.\.venv\Scripts\python.exe' -m PyInstaller --clean --noconfirm --distpath $DistPath --workpath $WorkPath Yapper.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed: $LASTEXITCODE" }
}
finally {
    $env:PATH = $previousPath
    Pop-Location
}
