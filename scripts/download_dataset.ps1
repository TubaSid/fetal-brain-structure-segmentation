param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [Parameter(Mandatory = $false)]
    [string[]]$Urls = @()
)

<#+
.SYNOPSIS
    Download and extract fetal ultrasound dataset archives.

.DESCRIPTION
    Given a list of direct archive URLs (e.g., from Zenodo record 10.5281/zenodo.8265464),
    this script downloads each file into $OutputDir and extracts it if it is a ZIP.

.EXAMPLE
    # 1) Get direct file URLs from https://doi.org/10.5281/zenodo.8265464
    # 2) Run script as:
    pwsh -File scripts/download_dataset.ps1 -OutputDir data -Urls "https://zenodo.org/records/8265464/files/TT.zip?download=1" "https://zenodo.org/records/8265464/files/TV.zip?download=1"

.NOTES
    - Use direct file links; the DOI page itself is HTML and not a dataset file.
    - Requires PowerShell 5+ on Windows (Expand-Archive available) or PowerShell 7.
#>

function Ensure-Dir($path) {
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
}

function Download-File($url, $destPath) {
    Write-Host "Downloading:" $url
    Invoke-WebRequest -Uri $url -OutFile $destPath -UseBasicParsing
}

function Try-Unzip($archivePath, $extractDir) {
    if ($archivePath.ToLower().EndsWith('.zip')) {
        Write-Host "Extracting:" $archivePath
        Ensure-Dir $extractDir
        Expand-Archive -LiteralPath $archivePath -DestinationPath $extractDir -Force
    } else {
        Write-Host "Skipping extract (not a .zip):" $archivePath
    }
}

# Main
Ensure-Dir $OutputDir
$archivesDir = Join-Path $OutputDir 'archives'
Ensure-Dir $archivesDir

if ($Urls.Count -eq 0) {
    Write-Warning "No URLs provided. Get direct links from the Zenodo record and pass them via -Urls."
    exit 1
}

$i = 0
foreach ($u in $Urls) {
    $i += 1
    $fileName = ([System.Uri]$u).Segments[-1]
    if ($fileName.Contains('?')) { $fileName = $fileName.Split('?')[0] }
    if ([string]::IsNullOrWhiteSpace($fileName)) { $fileName = "file_$i.zip" }

    $dest = Join-Path $archivesDir $fileName
    Download-File -url $u -destPath $dest
    Try-Unzip -archivePath $dest -extractDir $OutputDir
}

Write-Host "Done. Archives in: $archivesDir; Extracted under: $OutputDir"