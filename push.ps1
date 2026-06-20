# push.ps1 — clear git locks and push to GitHub
# Run this anytime after Claude makes changes: right-click → Run with PowerShell

Set-Location $PSScriptRoot

# Clear all lock files
Get-ChildItem .git -Filter "*.lock" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

# Stage, commit, push
git add -A
$msg = "chore: sync changes from Cowork"
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m $msg
    git push
    Write-Host "✓ Pushed successfully" -ForegroundColor Green
} else {
    Write-Host "Nothing to commit — already up to date" -ForegroundColor Yellow
}
