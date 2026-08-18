$dates = @(
    "2026-08-12",
    "2026-08-13",
    "2026-08-14",
    "2026-08-15",
    "2026-08-16",
    "2026-08-17",
    "2026-08-18"
)

$commitsPerDay = @(24, 12, 11, 26, 19, 21, 15) # Added 15 for the 7th day (Aug 18th)

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
    "Update requirements"
)

# Create a dummy file to modify so we have something to commit
$dummyFile = "progress_log.txt"
if (-not (Test-Path $dummyFile)) {
    New-Item -Path $dummyFile -ItemType File | Out-Null
    git add $dummyFile
    git commit -m "Initial commit for progress log" | Out-Null
}

for ($i = 0; $i -lt $dates.Length; $i++) {
    $currentDateStr = $dates[$i]
    $numCommits = $commitsPerDay[$i]
    
    Write-Host "Generating $numCommits commits for $currentDateStr..."
    
    for ($j = 0; $j -lt $numCommits; $j++) {
        # Random time between 18:00 and 01:00 next day
        # 18:00 to 23:59 (6 hours) or 00:00 to 00:59 (1 hour)
        $hourChoice = Get-Random -Minimum 0 -Maximum 7
        
        if ($hourChoice -eq 6) {
            # 00:00 to 00:59 next day
            $hour = 0
            $dateObj = (Get-Date $currentDateStr).AddDays(1)
        } else {
            # 18:00 to 23:59 same day
            $hour = 18 + $hourChoice
            $dateObj = Get-Date $currentDateStr
        }
        
        $minute = Get-Random -Minimum 0 -Maximum 60
        $second = Get-Random -Minimum 0 -Maximum 60
        
        $commitDate = $dateObj.AddHours($hour).AddMinutes($minute).AddSeconds($second)
        $isoDate = $commitDate.ToString("yyyy-MM-ddTHH:mm:ss")
        
        # Modify the file
        Add-Content -Path $dummyFile -Value "Commit generated at $isoDate"
        
        # Stage the file
        git add $dummyFile
        
        # Set environment variables for git
        $env:GIT_AUTHOR_DATE = $isoDate
        $env:GIT_COMMITTER_DATE = $isoDate
        
        # Commit
        $msg = $messages | Get-Random
        git commit -m $msg | Out-Null
    }
}

Write-Host "Done generating backdated commits. You can now run 'git push'."
