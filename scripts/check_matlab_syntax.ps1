# check_matlab_syntax.ps1
# Automated MATLAB Syntax and Delimiter Verification Script

$workspace = Split-Path -Parent $PSScriptRoot
if (-not $workspace) { $workspace = Get-Location }

$mFiles = Get-ChildItem -Path $workspace -Recurse -Filter '*.m' | Where-Object { $_.FullName -notmatch '\\.git\\' }
Write-Host "Found $($mFiles.Count) MATLAB (.m) files to validate." -ForegroundColor Cyan

$allErrors = @()

foreach ($file in $mFiles) {
    $filePath = $file.FullName
    $relPath = $filePath.Replace($workspace, "").TrimStart('\', '/')
    $lines = Get-Content -Path $filePath

    # 1. Check for backslash quote \" (MATLAB does not allow \" inside double quotes)
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        # Ignore comments
        $commentIdx = $line.IndexOf('%')
        $codePart = if ($commentIdx -ge 0) { $line.Substring(0, $commentIdx) } else { $line }
        if ($codePart -match '\\"') {
            $allErrors += [PSCustomObject]@{
                File = $relPath
                Line = ($i + 1)
                Type = 'ForbiddenBackslashQuote'
                Detail = $line.Trim()
            }
        }
    }

    # 2. Check balanced delimiters ignoring comments and strings
    $stack = New-Object System.Collections.Generic.Stack[char]
    $inBlockComment = $false
    $lineNum = 0

    foreach ($rawLine in $lines) {
        $lineNum++
        $trimmed = $rawLine.Trim()
        
        if ($inBlockComment) {
            if ($trimmed -match '^\%\}') {
                $inBlockComment = $false
            }
            continue
        }
        if ($trimmed -match '^\%\{') {
            $inBlockComment = $true
            continue
        }

        $len = $rawLine.Length
        $j = 0
        $prevChar = ' '

        while ($j -lt $len) {
            $c = $rawLine[$j]

            # Line comment
            if ($c -eq '%') {
                break
            }

            # Double-quoted string "..."
            if ($c -eq '"') {
                $j++
                while ($j -lt $len) {
                    if ($rawLine[$j] -eq '"') {
                        if (($j + 1) -lt $len -and $rawLine[$j + 1] -eq '"') {
                            $j += 2
                            continue
                        } else {
                            $j++
                            break
                        }
                    }
                    $j++
                }
                $prevChar = '"'
                continue
            }

            # Single-quoted string/character vector '...'
            # In MATLAB, ' is transpose if preceded by alphanumeric, ), ], }, or .
            # Otherwise it initiates a character vector
            $isTranspose = ($prevChar -match '[a-zA-Z0-9_\)\]\}\.]')
            if ($c -eq "'" -and -not $isTranspose) {
                $j++
                while ($j -lt $len) {
                    if ($rawLine[$j] -eq "'") {
                        if (($j + 1) -lt $len -and $rawLine[$j + 1] -eq "'") {
                            $j += 2
                            continue
                        } else {
                            $j++
                            break
                        }
                    }
                    $j++
                }
                $prevChar = "'"
                continue
            }

            if (-not [char]::IsWhiteSpace($c)) {
                $prevChar = $c
            }

            # Delimiters
            if ($c -eq '(' -or $c -eq '[' -or $c -eq '{') {
                $stack.Push($c)
            }
            elseif ($c -eq ')' -or $c -eq ']' -or $c -eq '}') {
                if ($stack.Count -eq 0) {
                    $allErrors += [PSCustomObject]@{
                        File = $relPath
                        Line = $lineNum
                        Type = 'UnmatchedClosingDelimiter'
                        Detail = "Found '$c' with empty stack at char $($j+1): $trimmed"
                    }
                } else {
                    $top = $stack.Pop()
                    $expected = switch ($top) {
                        '(' { ')' }
                        '[' { ']' }
                        '{' { '}' }
                    }
                    if ($c -ne $expected) {
                        $allErrors += [PSCustomObject]@{
                            File = $relPath
                            Line = $lineNum
                            Type = 'MismatchedDelimiter'
                            Detail = "Expected '$expected' for '$top' but found '$c' at char $($j+1): $trimmed"
                        }
                    }
                }
            }
            $j++
        }
    }

    if ($stack.Count -gt 0) {
        $remaining = -join ($stack.ToArray())
        $allErrors += [PSCustomObject]@{
            File = $relPath
            Line = $lineNum
            Type = 'UnclosedDelimitersAtEOF'
            Detail = "Unclosed stack: $remaining"
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "SYNTAX & DELIMITER VERIFICATION SUMMARY" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

if ($allErrors.Count -eq 0) {
    Write-Host "PASS: 0 syntax or delimiter errors found across all $($mFiles.Count) MATLAB files!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "FAIL: Found $($allErrors.Count) syntax / delimiter error(s):" -ForegroundColor Red
    $allErrors | Format-Table -AutoSize
    exit 1
}
