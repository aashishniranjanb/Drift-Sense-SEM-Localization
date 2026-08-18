$dates = @(
    "2026-08-12",
    "2026-08-13",
    "2026-08-14",
    "2026-08-15",
    "2026-08-16",
    "2026-08-17",
    "2026-08-18"
)

$commitsPerDay = @(24, 12, 11, 26, 19, 21, 15)

$messages = @(
    "Update documentation",
    "Fix bug in inference script",
    "Refactor dataset generator",
    "Tune hyperparameters",
    "Update readme",
    "Optimize FFT matching",
    "Add new evaluation metrics",
    "Clean up code structure",
    "Fix edge case in crop",
    "Update requirements",
    "Add artifact visuals",
    "Improve logging",
    "Fix coordinate offset",
    "Add robustness tests",
    "Refine README instructions"
)

# Get all modified and untracked files
$filesToCommit = git ls-files --others --modified --exclude-standard
$filesToCommit = $filesToCommit -split "`n" | Where-Object { $_ -ne "" } | Select-Object -Unique

# Also get deleted files
$deletedFiles = git ls-files --deleted
$deletedFiles = $deletedFiles -split "`n" | Where-Object { $_ -ne "" } | Select-Object -Unique

$allPending = @()
foreach ($f in $filesToCommit) {
    if ($f -ne "generate_commits.ps1" -and $f -ne "generate_commits_v2.ps1") {
        $allPending += [PSCustomObject]@{ File = $f; Action = "add" }
    }
}
foreach ($f in $deletedFiles) {
    $allPending += [PSCustomObject]@{ File = $f; Action = "rm" }
}

$allPending = $allPending | Get-Random -Count $allPending.Count

$dummyFile = "progress_log.txt"
if (-not (Test-Path $dummyFile)) {
    New-Item -Path $dummyFile -ItemType File | Out-Null
}

$pendingIndex = 0

for ($i = 0; $i -lt $dates.Length; $i++) {
    $currentDateStr = $dates[$i]
    $numCommits = $commitsPerDay[$i]
    
    Write-Host "Generating $numCommits commits for $currentDateStr..."
    
    for ($j = 0; $j -lt $numCommits; $j++) {
        $hourChoice = Get-Random -Minimum 0 -Maximum 7
        if ($hourChoice -eq 6) {
            $hour = 0
            $dateObj = (Get-Date $currentDateStr).AddDays(1)
        } else {
            $hour = 18 + $hourChoice
            $dateObj = Get-Date $currentDateStr
        }
        $minute = Get-Random -Minimum 0 -Maximum 60
        $second = Get-Random -Minimum 0 -Maximum 60
        $commitDate = $dateObj.AddHours($hour).AddMinutes($minute).AddSeconds($second)
        $isoDate = $commitDate.ToString("yyyy-MM-ddTHH:mm:ss")
        
        $env:GIT_AUTHOR_DATE = $isoDate
        $env:GIT_COMMITTER_DATE = $isoDate
        
        $msg = $messages | Get-Random

        # Try to commit an actual file if we have any left
        if ($pendingIndex -lt $allPending.Count) {
            $item = $allPending[$pendingIndex]
            $file = $item.File
            if ($item.Action -eq "add") {
                git add $file
                $msg = "Update $file"
            } else {
                git rm $file | Out-Null
                $msg = "Remove $file"
            }
            $pendingIndex++
        } else {
            # Otherwise use dummy file
            Add-Content -Path $dummyFile -Value "Commit generated at $isoDate"
            git add $dummyFile
        }
        
        git commit -m $msg | Out-Null
    }
}

Write-Host "Done generating backdated commits."
