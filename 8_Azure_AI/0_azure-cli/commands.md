`az login --use-device-code`

`az account subscription list`

Create service principal
`az ad sp create-for-rbac -n "api://<spName>" --role owner --scopes subscriptions/<subscriptionId>/resourceGroups/<resourceGroup>`
`az ad sp create-for-rbac -n "api://ai-sp" --role owner --scopes subscriptions/ee94704a-0e63-41fb-8477-02276718ba21/resourceGroups/Get-started-102-RG`
