@ECHO OFF

REM Command file for Sphinx documentation
REM SCPN-Fusion-Core Documentation Build
REM
REM Usage:
REM   make.bat html        Build HTML documentation
REM   make.bat clean       Remove build artifacts
REM   make.bat linkcheck   Check external links

pushd %~dp0

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help
if "%1" == "clean" goto clean

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
