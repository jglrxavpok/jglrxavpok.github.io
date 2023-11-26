$now = Get-Date

$title = Read-Host "Title?"
if($title.Length -eq 0) {
    exit 1;
}
$link = Read-Host "Link?"
if($link.Length -eq 0) {
    exit 2;
}

$filename = "_posts\$($now.Year)-$($now.Month)-$($now.Day)-$link.markdown";

$contents = @"
---
layout: post
title:  "$title"
date:   $($now.ToString("yyyy-MM-dd HH:mm:ss")) +0100
categories: TODO!
---
Todo: write post
"@

$contents | Set-Content -Path $filename