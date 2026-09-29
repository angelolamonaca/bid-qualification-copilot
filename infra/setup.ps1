<#
One-time Azure setup for the Bid Qualification Copilot. Safe to re-run.

Creates in resource group rg-bid-copilot (Switzerland North), next to the AI resource from Step 1:
  1. a user-assigned managed identity: the app's own identity (no passwords, no keys)
  2. a storage account for the workflow state, with key access DISABLED (identities only)
  3. a container registry for the app's image
  4. role assignments: exactly what the identity needs, each scoped to one resource
  5. a Container Apps environment (with a Log Analytics workspace for the logs)

Prerequisites: `az login` done, Step 1 completed. Run from the project root:  .\infra\setup.ps1
#>
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true   # a failing az or docker command stops the script (PowerShell 7.3+)
. "$PSScriptRoot\names.ps1"   # resource names shared by all scripts

Write-Host "0/5 Azure CLI extension for Container Apps"
az extension add --name containerapp --upgrade --only-show-errors

Write-Host "1/5 Managed identity '$identity': the app will sign in to Azure as this identity"
$id = az identity create --name $identity --resource-group $rg --location $location `
    --query "{principalId: principalId, clientId: clientId}" -o json | ConvertFrom-Json

Write-Host "2/5 Storage account '$storage' with shared-key access disabled (Entra identities only)"
az storage account create --name $storage --resource-group $rg --location $location `
    --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2 `
    --allow-blob-public-access false --allow-shared-key-access false --only-show-errors | Out-Null

Write-Host "3/5 Container registry '$registry' (Basic tier, about 0.17 USD per day)"
az acr create --name $registry --resource-group $rg --location $location --sku Basic `
    --only-show-errors | Out-Null

Write-Host "4/5 Roles for the app identity: least privilege, each on one resource"
$roles = [ordered]@{
    "AcrPull"                        = az acr show --name $registry --query id -o tsv
    "Storage Table Data Contributor" = az storage account show --name $storage --query id -o tsv
    "Cognitive Services OpenAI User" = az cognitiveservices account show --name $aiResource `
                                           --resource-group $rg --query id -o tsv
}
foreach ($role in $roles.Keys) {
    Write-Host "     - $role"
    az role assignment create --assignee-object-id $id.principalId `
        --assignee-principal-type ServicePrincipal --role $role --scope $roles[$role] `
        --only-show-errors | Out-Null
}

Write-Host "5/5 Container Apps environment '$environment' (takes a few minutes)"
az containerapp env create --name $environment --resource-group $rg --location $location `
    --only-show-errors | Out-Null

Write-Host "`nSetup done. Role assignments can take a few minutes to take effect."
