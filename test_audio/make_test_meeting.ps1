# Generates test_meeting.wav using Windows' built-in speech engine (no install, no internet).
# Run in PowerShell:  powershell -ExecutionPolicy Bypass -File .\make_test_meeting.ps1
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($s.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name })
$en = @($s.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like "en*" } | ForEach-Object { $_.VoiceInfo.Name })
if ($en.Count -eq 0) { $en = $voices }
Write-Host "Using voices:" ($en -join ", ")
$out = Join-Path (Get-Location) "test_meeting.wav"
$s.SetOutputToWaveFile($out)
$s.Rate = 0

$lines = @(
 "0|Okay everyone, let's start the sprint planning meeting. Today we need to cover the database, deployment, and the machine learning service.",
 "1|Sure. First, the database. We have been running on MySQL, but the reporting queries are getting slow. I suggest we move to PostgreSQL.",
 "0|Does anyone object to PostgreSQL?",
 "2|No objection from me. It has better support for JSON and window functions.",
 "0|Great, then it is decided. We will migrate to PostgreSQL. Priya, can you write the migration script?",
 "1|Yes, I will write the migration script and have it ready by Friday.",
 "0|Perfect. Next, deployment. Arjun, where are we with the container setup?",
 "2|The services are containerized. I think we should deploy everything on Kubernetes instead of the old virtual machines.",
 "0|Agreed. We are going with Kubernetes for production. Arjun, please set up the CI and CD pipeline for it.",
 "2|Okay, I will set up the CI CD pipeline. I can't promise a date yet because I need to check the cluster quota first.",
 "1|Maybe we could also try GraphQL for the new public API. It might reduce the number of requests.",
 "0|That is an interesting idea, but we are not ready to commit. Let's revisit GraphQL in the next quarter.",
 "2|Also, somebody should probably look at adding Redis caching, because the dashboard is slow at peak hours.",
 "0|True, but nobody has time this sprint. We will leave Redis for now.",
 "0|Now the machine learning service. Meera, how did the PyTorch model benchmarks go?",
 "3|The PyTorch model reaches ninety two percent accuracy, but inference takes four hundred milliseconds, which is too slow. I will share the full benchmark results with the team by next Tuesday.",
 "0|Thanks Meera. One more thing. We are not going to upgrade to OAuth version two this sprint, because the security review is still open.",
 "1|Understood. That is a decision then, no OAuth upgrade this sprint.",
 "0|Right. To summarize, we are moving to PostgreSQL, deploying on Kubernetes, and postponing GraphQL, Redis, and the OAuth upgrade. Thanks everyone, meeting adjourned."
)
foreach ($l in $lines) {
  $i, $t = $l.Split("|", 2)
  $s.SelectVoice($en[[int]$i % $en.Count])
  $s.Speak($t)
  $s.SelectVoice($en[0]); $s.Speak(" ")
}
$s.Dispose()
Write-Host "Created $out"
