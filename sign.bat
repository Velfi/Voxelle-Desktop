@echo off
relic sign --file "%~1" --key azure --config "%~dp0src-tauri\relic.conf"
