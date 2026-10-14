@echo off
setlocal
set "TOOLKIT_ROOT=%~dp0..\.."
where conda >nul 2>nul
if errorlevel 1 (
  echo ERROR: conda is not available in PATH 1>&2
  exit /b 1
)
for %%C in (core surya marker) do (
  call :ensure_env %%C digital-pdf-%%C
  if errorlevel 1 exit /b 1
)
call conda run -n digital-pdf-core python -m pip install -e "%TOOLKIT_ROOT%"
if errorlevel 1 exit /b 1
exit /b 0

:ensure_env
set "COMPONENT=%~1"
set "ENV_NAME=%~2"
call conda env list | findstr /B /C:"%ENV_NAME% " >nul
if errorlevel 1 (
  call conda env create --file "%TOOLKIT_ROOT%\environments\%COMPONENT%\environment.yml"
) else (
  call conda env update --name "%ENV_NAME%" --file "%TOOLKIT_ROOT%\environments\%COMPONENT%\environment.yml" --prune
)
if errorlevel 1 exit /b 1
call conda run -n "%ENV_NAME%" python -m pip install -r "%TOOLKIT_ROOT%\environments\%COMPONENT%\requirements.txt"
if errorlevel 1 exit /b 1
exit /b 0
