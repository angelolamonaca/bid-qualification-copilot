<#
Build the image on this machine, push it to the registry, then create or update the container app.
Run after setup.ps1, and again after every code change: each run creates a new revision, named
after the git commit it was built from, so every deployment is traceable to its code.

Run from the project root:  .\infra\deploy.ps1
#>
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true   # a failing az or docker command stops the script (PowerShell 7.3+)
. "$PSScriptRoot\names.ps1"

$tag = git rev-parse --short HEAD
$image = "$registry.azurecr.io/$($app):$tag"

Write-Host "1/3 Build the image $image (the same image that runs locally)"
docker build -t $image .
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

Write-Host "2/3 Push it to the registry"
az acr login --name $registry --only-show-errors
docker push $image
if ($LASTEXITCODE -ne 0) { throw "docker push failed" }

Write-Host "3/3 Create or update the container app '$app'"
$id = az identity show --name $identity --resource-group $rg --query "{id: id, clientId: clientId}" `
    -o json | ConvertFrom-Json
# Configuration only, no secrets: the app signs in to the model and to storage as its identity.
$settings = @(
    "APP_ENV=azure",
    "AZURE_OPENAI_ENDPOINT=https://$aiResource.openai.azure.com/openai/v1/",
    "CHAT_DEPLOYMENT=gpt-5-mini",
    "EMBEDDING_DEPLOYMENT=text-embedding-3-small",
    "STATE_BACKEND=table",
    "TABLES_ENDPOINT=https://$storage.table.core.windows.net",
    "CRM_BASE_URL=http://localhost:8000/mock-crm",
    "AZURE_CLIENT_ID=$($id.clientId)"   # tells DefaultAzureCredential which identity to use
)
$exists = az containerapp list --resource-group $rg --query "[?name=='$app'].name" -o tsv
if ($exists) {
    az containerapp update --name $app --resource-group $rg --image $image `
        --set-env-vars $settings --only-show-errors | Out-Null
} else {
    # min 0 replicas: scales to zero when idle (no cost), max 1: a demo needs no more.
    az containerapp create --name $app --resource-group $rg --environment $environment `
        --image $image --registry-server "$registry.azurecr.io" --registry-identity $id.id `
        --user-assigned $id.id --target-port 8000 --ingress external `
        --cpu 0.5 --memory 1.0Gi --min-replicas 0 --max-replicas 1 `
        --env-vars $settings --only-show-errors | Out-Null
}
$url = az containerapp show --name $app --resource-group $rg `
    --query properties.configuration.ingress.fqdn -o tsv
Write-Host "`nDeployed $tag -> https://$url"
