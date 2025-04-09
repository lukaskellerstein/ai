# Install

https://hub.docker.com/r/microsoft/mssql-server

```bash
docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=Heslo_1234" -p 1433:1433 -d mcr.microsoft.com/mssql/server:2022-latest
```

username: sa
password: viz docker command

# UI

Windows: MSSQL Management Studio

https://learn.microsoft.com/en-us/ssms/download-sql-server-management-studio-ssms

Mac: Azure Data Studio

https://learn.microsoft.com/en-us/azure-data-studio/download-azure-data-studio?view=sql-server-ver16&tabs=win-install%2Cwin-user-install%2Credhat-install%2Cwindows-uninstall%2Credhat-uninstall
