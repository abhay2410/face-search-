Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & "start_dashboard.bat" & chr(34), 0
Set WshShell = Nothing
