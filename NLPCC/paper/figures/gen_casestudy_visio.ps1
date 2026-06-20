param(
    [string]$DataPath = (Join-Path $PSScriptRoot 'data\casestudy_node11654_evidence.json'),
    [string]$VsdxPath = (Join-Path $PSScriptRoot 'casestudy.vsdx'),
    [string]$OutputDir = $PSScriptRoot,
    [string[]]$ExportFormats = @('png', 'svg', 'pdf'),
    [switch]$KeepVisioOpen
)

$ErrorActionPreference = 'Stop'
$PageW = 14.0
$PageH = 7.8
$RefW = 1400.0
$RefH = 780.0

function VX([double]$x) { $PageW * $x / $RefW }
function VY([double]$y) { $PageH - ($PageH * $y / $RefH) }
function RGBF([int]$r, [int]$g, [int]$b) { "RGB($r,$g,$b)" }

$C = @{
    Black = RGBF 24 34 48
    Gray = RGBF 102 112 133
    Panel = RGBF 248 250 252
    Blue = RGBF 76 120 168
    BlueSoft = RGBF 223 241 255
    Bot = RGBF 200 76 76
    Human = RGBF 76 120 168
    Green = RGBF 47 133 90
    GreenSoft = RGBF 236 248 239
    Orange = RGBF 217 130 43
    OrangeSoft = RGBF 255 245 217
    Peach = RGBF 251 231 217
    White = RGBF 255 255 255
    Line = RGBF 52 64 84
    PanelEdge = RGBF 154 165 177
}

function Set-Cell($shape, [string]$cell, [string]$formula) {
    try { $shape.CellsU($cell).FormulaU = $formula } catch {}
}

function Style-Shape($shape, [string]$fill, [string]$line, [double]$linePt = 0.8, [int]$dash = 1, [double]$roundPx = 6) {
    Set-Cell $shape 'FillPattern' '1'
    Set-Cell $shape 'FillForegnd' $fill
    Set-Cell $shape 'LinePattern' ([string]$dash)
    Set-Cell $shape 'LineColor' $line
    Set-Cell $shape 'LineWeight' "$linePt pt"
    if ($roundPx -gt 0) {
        Set-Cell $shape 'Rounding' ((VX $roundPx).ToString([Globalization.CultureInfo]::InvariantCulture) + ' in')
    }
}

function Set-Text($shape, [string]$text, [double]$size = 9, [string]$color = $C.Black, [bool]$bold = $false, [int]$align = 0) {
    $shape.Text = $text
    Set-Cell $shape 'Char.Size' "$size pt"
    Set-Cell $shape 'Char.Color' $color
    Set-Cell $shape 'Char.Style' ($(if ($bold) { '1' } else { '0' }))
    Set-Cell $shape 'Para.HorzAlign' ([string]$align)
    Set-Cell $shape 'VerticalAlign' '0'
    foreach ($m in 'TxtMarginLeft','TxtMarginRight','TxtMarginTop','TxtMarginBottom') {
        Set-Cell $shape $m '2 pt'
    }
}

function RectTL([double]$x, [double]$y, [double]$w, [double]$h, [string]$text = '', [string]$fill = $C.White, [string]$line = $C.Line, [double]$size = 9, [bool]$bold = $false, [double]$linePt = 0.8, [int]$dash = 1, [double]$roundPx = 6) {
    $s = $script:Page.DrawRectangle((VX $x), (VY ($y + $h)), (VX ($x + $w)), (VY $y))
    Style-Shape $s $fill $line $linePt $dash $roundPx
    if ($text -ne '') { Set-Text $s $text $size $C.Black $bold 0 }
    return $s
}

function TextTL([double]$x, [double]$y, [double]$w, [double]$h, [string]$text, [double]$size = 9, [string]$color = $C.Black, [bool]$bold = $false, [int]$align = 0) {
    $s = $script:Page.DrawRectangle((VX $x), (VY ($y + $h)), (VX ($x + $w)), (VY $y))
    Set-Cell $s 'FillPattern' '0'
    Set-Cell $s 'LinePattern' '0'
    Set-Text $s $text $size $color $bold $align
    return $s
}

function OvalTL([double]$x, [double]$y, [double]$w, [double]$h, [string]$text = '', [string]$fill = $C.White, [string]$line = $C.Line, [double]$size = 8, [bool]$bold = $false) {
    $s = $script:Page.DrawOval((VX $x), (VY ($y + $h)), (VX ($x + $w)), (VY $y))
    Style-Shape $s $fill $line 0.8 1 0
    if ($text -ne '') { Set-Text $s $text $size $C.White $bold 1 }
    return $s
}

function LineTL([double]$x1, [double]$y1, [double]$x2, [double]$y2, [string]$color = $C.Line, [double]$linePt = 0.8, [bool]$arrowEnd = $false, [int]$dash = 1) {
    $s = $script:Page.DrawLine((VX $x1), (VY $y1), (VX $x2), (VY $y2))
    Set-Cell $s 'LineColor' $color
    Set-Cell $s 'LineWeight' "$linePt pt"
    Set-Cell $s 'LinePattern' ([string]$dash)
    if ($arrowEnd) { Set-Cell $s 'EndArrow' '4' }
    return $s
}

function LabelColor([string]$label) {
    if ($label -eq 'Bot') { return $C.Bot }
    return $C.Human
}

function DrawNode([double]$cx, [double]$cy, [string]$label, [string]$color, [double]$r = 22, [string]$sub = '') {
    OvalTL ($cx - $r) ($cy - $r) (2 * $r) (2 * $r) $label $color $C.White 8 $true | Out-Null
    if ($sub -ne '') { TextTL ($cx - 32) ($cy + $r + 5) 64 18 $sub 6.5 $C.Gray $false 1 | Out-Null }
}

function DrawConfidence([double]$x, [double]$y, $branch) {
    $edge = $(if ($branch.correct) { $C.Green } else { $C.Bot })
    RectTL $x $y 225 88 '' $C.White $edge 8 $false 1.2 1 8 | Out-Null
    TextTL ($x + 12) ($y + 10) 125 20 ([string]$branch.name) 7.8 $C.Black $true 0 | Out-Null
    TextTL ($x + 142) ($y + 10) 70 20 ("conf " + ([double]$branch.true_confidence).ToString('0.000')) 6.7 $C.Gray $false 2 | Out-Null
    TextTL ($x + 12) ($y + 38) 70 18 'prediction' 6.6 $C.Gray $false 0 | Out-Null
    TextTL ($x + 92) ($y + 38) 70 18 ([string]$branch.prediction) 7.4 (LabelColor $branch.prediction) $true 0 | Out-Null
    TextTL ($x + 142) ($y + 38) 70 18 ($(if ($branch.correct) { 'correct' } else { 'wrong' })) 7.0 $edge $true 2 | Out-Null
    TextTL ($x + 12) ($y + 64) 190 16 ("margin true-other " + ([double]$branch.margin_true_minus_other).ToString('+0.000;-0.000')) 6.5 $C.Gray $false 0 | Out-Null
}

function ShortText([string]$text, [int]$limit = 120) {
    $clean = ($text -replace '\s+', ' ').Trim()
    if ($clean.Length -le $limit) { return $clean }
    return $clean.Substring(0, $limit - 3) + '...'
}

function Draw-CaseStudy($data) {
    RectTL 25 45 335 690 '' (RGBF 247 251 255) $C.PanelEdge 9 $false 1.0 1 10 | Out-Null
    RectTL 385 45 285 690 '' (RGBF 250 250 247) $C.PanelEdge 9 $false 1.0 1 10 | Out-Null
    RectTL 695 45 280 690 '' (RGBF 248 251 248) $C.PanelEdge 9 $false 1.0 1 10 | Out-Null

    TextTL 45 66 280 28 'Target Evidence' 10 $C.Black $true 0 | Out-Null
    TextTL 400 66 260 28 'Local / Support Neighborhood' 10 $C.Black $true 0 | Out-Null
    TextTL 725 66 220 28 'Branch Evidence' 10 $C.Black $true 0 | Out-Null

    $target = $data.target
    $case = $data.case
    $header = "node $($case.node_index) | $($target.user_id) | @$($target.username)`n" +
              "gold: $($case.gold_name)    risk: $(([double]$case.risk_score).ToString('0.000'))    routed"
    RectTL 45 115 295 65 $header $C.White $C.Bot 8 $true 1.0 1 8 | Out-Null

    RectTL 45 205 285 110 ("Description:`n" + (ShortText $target.description 120)) $C.OrangeSoft (RGBF 197 159 63) 6.8 $true 0.8 1 6 | Out-Null
    $meta = "Metadata:`ncreated: $($target.created_at.Substring(0, [Math]::Min(16, $target.created_at.Length)))`n" +
            "followers/following: $($target.followers_count) / $($target.following_count)`n" +
            "tweets/listed: $($target.tweet_count) / $($target.listed_count)`n" +
            "verified/protected: $($target.verified) / $($target.protected)"
    RectTL 45 335 285 125 $meta $C.Peach (RGBF 201 137 98) 6.9 $true 0.8 1 6 | Out-Null

    $tweetLines = @()
    for ($i = 0; $i -lt [Math]::Min(4, $target.tweets.Count); $i++) {
        $tweetLines += ("{0}. {1}" -f ($i + 1), (ShortText ([string]$target.tweets[$i]) 88))
    }
    RectTL 45 485 285 220 ("Tweets:`n" + ($tweetLines -join "`n")) $C.BlueSoft (RGBF 106 158 195) 6.5 $true 0.8 1 6 | Out-Null

    $cx = 525.0
    $cy = 425.0
    DrawNode $cx $cy 'T' $C.Bot 32 '11654'
    TextTL 490 350 70 18 'target' 7 $C.Gray $false 1 | Out-Null

    $relPos = @(@(455, 255), @(600, 255))
    for ($i = 0; $i -lt $data.relation_neighbors.Count; $i++) {
        $nb = $data.relation_neighbors[$i]
        $pos = $relPos[$i]
        LineTL $cx $cy $pos[0] $pos[1] $C.Line 1.0 $false 1 | Out-Null
        DrawNode $pos[0] $pos[1] ([string]$nb.label_name).Substring(0,1) (LabelColor $nb.label_name) 22 ([string]$nb.node_index)
    }
    TextTL 435 190 190 20 ("relation bot ratio = " + ([double]$case.relation_bot_ratio).ToString('0.000')) 7.5 $C.Black $false 1 | Out-Null

    $supportPos = @(@(430,535),@(465,595),@(515,625),@(575,605),@(620,545),@(610,465),@(555,492),@(475,492))
    for ($i = 0; $i -lt $data.support_neighbors.Count; $i++) {
        $nb = $data.support_neighbors[$i]
        $pos = $supportPos[$i]
        LineTL $cx $cy $pos[0] $pos[1] $C.Gray 0.8 $false 2 | Out-Null
        DrawNode $pos[0] $pos[1] ([string]$nb.label_name).Substring(0,1) (LabelColor $nb.label_name) 18 ([string]$nb.rank)
    }
    TextTL 420 650 210 18 ("KNN support bot ratio = " + ([double]$case.support_bot_ratio).ToString('0.000')) 7.5 $C.Black $false 1 | Out-Null
    TextTL 430 680 190 18 'support labels: B B B B B H B B' 7.2 $C.Gray $false 1 | Out-Null

    $y = 125.0
    foreach ($branch in $data.branch_evidence) {
        DrawConfidence 725 $y $branch
        $y += 110
    }
    RectTL 725 590 225 90 "Final decision:`nRouted-only residual -> Bot`nLow-order error is corrected." $C.White $C.Green 6.9 $true 1.2 1 8 | Out-Null
}

if (-not (Test-Path -LiteralPath $DataPath)) {
    throw "Data JSON not found: $DataPath"
}
$data = Get-Content -LiteralPath $DataPath -Raw -Encoding UTF8 | ConvertFrom-Json

$visio = $null
$doc = $null
try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $false

    if (Test-Path -LiteralPath $VsdxPath) {
        $backup = Join-Path (Split-Path -Parent $VsdxPath) (([IO.Path]::GetFileNameWithoutExtension($VsdxPath)) + ".backup-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + ".vsdx")
        Copy-Item -LiteralPath $VsdxPath -Destination $backup
        Write-Output "Backup: $backup"
    }

    $doc = $visio.Documents.Add('')
    $script:Page = $doc.Pages.Item(1)
    $script:Page.PageSheet.CellsU('PageWidth').FormulaU = "$PageW in"
    $script:Page.PageSheet.CellsU('PageHeight').FormulaU = "$PageH in"
    Draw-CaseStudy $data
    $doc.SaveAs($VsdxPath) | Out-Null

    foreach ($formatRaw in $ExportFormats) {
        foreach ($format in ($formatRaw -split ',')) {
            $name = $format.Trim().TrimStart('.').ToLowerInvariant()
            if (-not $name) { continue }
            $outPath = Join-Path $OutputDir ("casestudy.$name")
            switch ($name) {
                'png' { $script:Page.Export($outPath) }
                'svg' { $script:Page.Export($outPath) }
                'pdf' { $doc.ExportAsFixedFormat(1, $outPath, 1, 0) }
                default { throw "Unsupported export format: $name" }
            }
            Write-Output ("{0}: {1} ({2} bytes)" -f $name.ToUpperInvariant(), $outPath, (Get-Item -LiteralPath $outPath).Length)
        }
    }
    Write-Output "Saved: $VsdxPath"
} finally {
    if ($doc -ne $null) { try { $doc.Close() } catch {} }
    if (($visio -ne $null) -and (-not $KeepVisioOpen)) { try { $visio.Quit() } catch {} }
}
