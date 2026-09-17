@echo off
set "PATH=%LOCALAPPDATA%\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links;%LOCALAPPDATA%\Programs\Python\Python314\Scripts;%PATH%"
where uv
uv --version
mcp dev weather_server.py
