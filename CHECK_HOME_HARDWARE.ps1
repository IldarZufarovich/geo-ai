Write-Host "=== CPU ==="
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
Write-Host "=== RAM ==="
Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum | Select-Object @{N='RAM_GB';E={[math]::Round($_.Sum/1GB,1)}}
Write-Host "=== GPU ==="
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM
Write-Host "=== DISKS ==="
Get-PhysicalDisk | Select-Object FriendlyName, MediaType, @{N='Size_GB';E={[math]::Round($_.Size/1GB,0)}}
Write-Host "=== NVIDIA (if available) ==="
try { nvidia-smi } catch { Write-Host "nvidia-smi not available" }
