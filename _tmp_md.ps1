$nb=Get-Content -Raw 'Uganda_Grain_SARIMA_Analysis.ipynb' | ConvertFrom-Json  
$md=$nb.cells | Where-Object { $_.cell_type -eq 'markdown' }  
$limit=[Math]::Min(3,$md.Count)  
for($i=0;$i -lt $limit;$i++){ $md[$i].source -join '' ; '---' } 
