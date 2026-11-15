@echo off
cd /d "C:\Users\rjwil\LocalDocuments\GitProjects\FC Macros"
git subtree push --prefix=AGV_Rover github main
pause

rem git subtree split --prefix=AGV_Rover -b agv-rover-public 
rem git push -f github agv-rover-public:main 
rem git branch -D agv-rover-public
