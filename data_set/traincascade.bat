opencv_traincascade.exe -data haarcascade -vec pos.vec -bg neg.txt -numPos 550-numNeg 2001 -numStages 20 -w 30 -h 30 -minHitRate 0.9 -maxFalseAlarmRate 0.5 -mode ALL
pause