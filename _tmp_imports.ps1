$nb=Get-Content -Raw 'Uganda_Grain_SARIMA_Analysis.ipynb' | ConvertFrom-Json  
$imports=@()  
foreach($c in $nb.cells){ if($c.cell_type -eq 'code'){ foreach($line in $c.source){ if($line -match 'import ' -or $line -match 'from '){ $imports += $line.Trim() } } } }  
$imports | Sort-Object -Unique 
