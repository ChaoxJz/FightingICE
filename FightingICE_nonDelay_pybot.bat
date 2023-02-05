::rem batファイルサンプル

setlocal ENABLEDELAYEDEXPANSION
::-Xmx8240m -Xms8240m -Xmn5120m 
:: close CUDA code
set CUDA_VISIBLE_DEVICES=100
java -cp FightingICE_nonDelay.jar;./lib/lwjgl/*;./lib/natives/windows/*;./lib/*;  Main  --py4j --limithp 400 400
rem TIMEOUT /T -1
endlocal
pause
exit