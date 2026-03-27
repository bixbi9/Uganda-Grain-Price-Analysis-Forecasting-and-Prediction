$nb=Get-Content -Raw 'Uganda_Grain_SARIMA_Analysis.ipynb' | ConvertFrom-Json  
$heads=@()  
foreach($c in $nb.cells){ if($c.cell_type -eq 'markdown'){ foreach($line in $c.source){ if($line -match '\s*#'){ $heads += $line.Trim() } } } }  
$heads 
