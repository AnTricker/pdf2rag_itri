@echo off
conda run --no-capture-output -n digital-pdf-core python -m digital_pdf_toolkit.cli extract %*
exit /b %ERRORLEVEL%
