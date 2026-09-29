<#
Delete EVERYTHING created for this project: the resource group and all resources in it,
including the AI resource from Step 1. Irreversible. Run from the project root:  .\infra\cleanup.ps1
#>
. "$PSScriptRoot\names.ps1"

$answer = Read-Host "Delete resource group '$rg' and everything in it? Type the group name to confirm"
if ($answer -ne $rg) { Write-Host "Cancelled."; exit }

az group delete --name $rg --yes

# The AI resource is soft-deleted and keeps its name for 48 hours. Purge it to free the name now:
az cognitiveservices account purge --name $aiResource --resource-group $rg --location $location

Write-Host @"
Deleted. Two things live outside the resource group, remove them in the portal if you created them:
  - the Entra app registration created for sign-in (Microsoft Entra ID > App registrations)
  - the budget alert (Cost Management > Budgets)
"@
